"""
Module 2: 6-state autonomous drilling controller + dual-environment mission simulation.

Discrete-time (dt = 1 s) simulation of IDLE -> ACTIVE -> [SEARCH -> DRILL -> RETRACTION
-> TRANSFER] x bites, driven only by the physics in src/terramechanics.py, the design
derivations in src/design.py and the solar model in src/solar.py. Nothing is scripted:
faults happen only if the bit thermal model or the auger torque model crosses a limit.

Controller laws (design choices, documented on the website):
  * WOB servo        : hold the derived set-point m g / FOS
  * Mode selection   : each step, evaluate rotary-only and rotary-percussive at the commanded
                       WOB/RPM and choose the mode with the lower mechanical specific energy
                       (percussion only where it actually helps)
  * Thermal derating : RPM = N_nom x clip((T_crit - T_bit) / 30 K, 0.4, 1)
  * Thermal fault    : T_bit >= T_crit  -> lift 3 cm off bottom, stop, dwell until
                       T_bit <= T_crit - margin (report Sec. 4.3); trip out if it cannot cool
  * Torque fault     : tau >= 0.85 tau_max -> trip out, brush, resume
  * Bite end         : 10 cm reached -> trip out (no rotation, cuttings stay in the flutes),
                       TRANSFER sweeps the sampling flutes past the brush while rotating

Bit thermal model:
  C_bit dT/dt = chi P_face - (G_wall + G_cuttings) (T - T_amb)        in the hole
  G_wall = k_layer S,  S = 2 pi L / ln(2L/r)  (cylinder shape factor)
  G_cuttings = phi rho A ROP c_p                  (fresh cuttings carry heat away)
  out of the hole: radiation eps sigma A (T^4 - T_sky^4) (+ convection on Mars)

Exports: telemetry_<env>.csv, kinematic_keyframes.json, abcd_summary.json
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.design import EXPORTS_DATA_DIR, PROJECT_ROOT, SIGMA_SB, derive_all, load_system_params
from src.solar import CONFIG_LABELS, array_power, trade_study
from src.terramechanics import Physics

DOCS_ASSETS_DATA_DIR = PROJECT_ROOT / "docs_src" / "assets" / "data"
STATES = ("IDLE", "ACTIVE", "SEARCH", "DRILL", "RETRACTION", "TRANSFER")


def simulate(p: Dict[str, Any], d: Dict[str, Any], env_id: str, solar_cfg: str) -> pd.DataFrame:
    ph = Physics(p, d, env_id)
    env = p["environments"][env_id]
    feed = p["actuators"]["z_axis_feed"]
    rot = p["actuators"]["rotary"]
    av = p["actuators"]["avionics_and_thermal"]
    cm = p["cutting_model"]
    stor = p["power_system"]["energy_storage"]
    crit = p["criteria"]
    lay = p["cad_layout"]

    dt = float(p["system"]["dt_s"])
    bite = float(p["system"]["bite_depth_m"])
    target = float(p["system"]["target_depth_m"])
    wob_set = float(d["wob"][env_id]["wob_setpoint_n"])
    wob_lim = float(d["wob"][env_id]["wob_limit_n"])
    rpm_nom = float(rot["nominal_rpm"])
    rpm_min = float(rot["min_rpm"])
    tau_fault = float(d["torque_fault_nm"])
    t_crit = float(d["critical_temp_k"][env_id])
    t_resume = t_crit - float(crit["thermal_resume_margin_k"])
    t_amb = float(env["initial_surface_temp_k"])
    t_sky = float(env["sky_temp_k"])
    c_bit = float(d["bit_thermal_mass_j_k"])
    chi = float(cm["heat_to_bit_fraction"])
    phi_c = float(cm["cuttings_heat_contact"])
    l_c = float(cm["bit_conduction_length_m"])
    shape = 2 * math.pi * l_c / math.log(2 * l_c / ph.r_bit)
    a_bit_rad = 2 * math.pi * ph.r_bit * l_c
    h_conv = float(av["mars_convection_coeff_w_m2k"]) if env_id == "mars" else 0.0
    ctl = p["controller"]
    idle_contact = float(cm["idle_wall_contact_fraction"])
    eps_bit = float(cm["bit_emissivity"])
    t_brush = float(d["brush"]["drag"]["total_nm"])        # derived (src/design_brush.py): flutes full
    t_brush_idle = float(d["brush"]["drag"]["idle_nm"])    # empty flights passing the brush while drilling
    derate_band = float(ctl["thermal_derate_band_k"])
    derate_floor = float(ctl["rpm_derate_floor"])
    standoff = float(ctl["thermal_standoff_m"])
    max_dwell = int(ctl["max_cooling_dwell_s"])

    v_search = float(feed["search_speed_m_s"])
    v_fast = float(feed["retraction_speed_m_s"])
    clear = float(feed["fast_approach_clearance_m"])
    w_contact = float(feed["contact_detect_wob_n"])
    m_move = float(feed["moving_mass_kg"])
    lead = float(feed["lead_mm"]) / 1e3
    d_m = float(feed["screw_mean_diameter_mm"]) / 1e3
    lam = math.atan(lead / (math.pi * d_m))
    phi_t = math.atan(float(feed["thread_friction_coeff"]) / math.cos(math.radians(15)))
    eta_screw = math.tan(lam) / math.tan(lam + phi_t)
    eta_feed = eta_screw * float(feed["motor_efficiency"])
    p_feed_nl = float(feed["no_load_power_w"])
    p_avio = float(av["base_avionics_power_w"])
    p_heat = float(av["heater_power_w"])

    # Kinematic positions (z = depth of bit tip below surface; negative = above surface)
    aug = d["auger"]
    zone0 = 0.026
    zone1 = zone0 + aug["sampling_length_mm"] / 1e3
    brush_h = float(lay["brush_axle_height_m"])
    z_home = -float(lay["home_tip_height_m"])
    z_sweep_start = -(brush_h - zone1 - 0.01)       # sampling-zone top just below the brush
    z_sweep_end = -(brush_h - zone0 + 0.01)         # sampling-zone bottom just above the brush
    v_sweep = aug["brush_sweep_mm_s"] / 1e3
    rpm_tr = float(p["actuators"]["brush_station"]["transfer_rpm"])

    cap_wh = float(stor["nominal_capacity_wh"])
    e_bat = cap_wh  # start full
    e_floor = cap_wh * (1 - float(stor["depth_of_discharge_limit"]))
    eta_ch = float(stor["charge_efficiency"])

    # Sample carousel: one station move (cycloidal) at the start of each bite's SEARCH and before the final stow,
    # in parallel with the feed; it only adds its index power (src/design_brush.py)
    car = d["brush"]["carousel"]
    t_ix, p_ix = float(car["index_time_s"]), float(car["index_power_w"])

    rec: List[Dict[str, Any]] = []
    st = {"t": 0.0, "z": z_home, "hole": 0.0, "T": t_amb, "bite": 0, "station": 0.0, "ix_from": 0.0, "ix_t": t_ix}

    def start_index() -> None:
        st["ix_from"] = round(st["station"])
        st["ix_t"] = 0.0

    def feed_power(force_n: float, v: float) -> float:
        return p_feed_nl + abs(force_n * v) / eta_feed if v != 0 else 0.0

    def bit_cooling(z: float, rop: float, rho: float, cp: float, k: float) -> float:
        T = st["T"]
        if z > 0.005:
            g = k * shape * (1.0 if rop > 0 else idle_contact) + phi_c * rho * ph.a_hole * rop * cp
            return g * (T - t_amb)
        return eps_bit * SIGMA_SB * a_bit_rad * (T ** 4 - t_sky ** 4) + h_conv * a_bit_rad * (T - t_amb)

    def log(state: str, wob=0.0, rpm=0.0, torque=0.0, perc=0, p_drill=0.0, p_feed_w=0.0, heater=0.0,
            mse=0.0, rop=0.0, fault="NONE", brush=0, q_in=0.0, mode="-", mse_alt=0.0) -> None:
        nonlocal e_bat
        L = ph.layer(max(st["z"], 0.0))
        cool = bit_cooling(st["z"], rop, float(L["density_kg_m3"]), float(L["specific_heat_j_kgk"]),
                           float(L["thermal_conductivity_w_mk"]))
        st["T"] += (q_in - cool) / c_bit * dt
        sol = array_power(solar_cfg, p, env_id, env, st["t"], geo=d["mast"]["solar"])   # nominal hardware
        p_car = 0.0
        if st["ix_t"] < t_ix:
            st["ix_t"] = min(t_ix, st["ix_t"] + dt)
            u = st["ix_t"] / t_ix
            st["station"] = st["ix_from"] + u - math.sin(2 * math.pi * u) / (2 * math.pi)
            p_car = p_ix
        p_load = p_avio + p_drill + p_feed_w + heater + p_car
        net = sol["power_w"] - p_load
        e_bat = min(cap_wh, e_bat + (net * eta_ch if net > 0 else net) * dt / 3600.0)
        rec.append({
            "time_s": st["t"], "environment": env_id, "state": state, "bite_number": st["bite"],
            "depth_m": round(st["z"], 4), "depth_cm": round(st["z"] * 100, 2),
            "hole_bottom_m": round(st["hole"], 4), "hole_bottom_cm": round(st["hole"] * 100, 2),
            "layer_name": L["name"], "ucs_mpa": round(ph.layer(max(st["z"], 0.0)) and ph.cutting(max(st["z"], 0.0), 0.0, 0.0, False)["ucs_pa"] / 1e6, 4),
            "wob_n": round(wob, 2), "wob_max_n": round(wob_lim, 2), "rpm": round(rpm, 1),
            "torque_nm": round(torque, 4), "percussion_duty": perc, "drill_mode": mode,
            "bit_temp_k": round(st["T"], 2), "crit_temp_k": t_crit,
            "power_w": round(p_load, 2), "solar_w": round(sol["power_w"], 1),
            "sun_az_deg": round(sol["az_deg"], 2), "sun_el_deg": round(sol["elev_deg"], 3),
            "battery_wh": round(e_bat, 2), "mse_mj_m3": round(mse / 1e6, 3) if np.isfinite(mse) else 0.0,
            # MSE of the mode the controller rejected (0 = not drilling, or that mode cannot penetrate)
            "mse_alt_mj_m3": round(mse_alt / 1e6, 3) if np.isfinite(mse_alt) else 0.0,
            "rop_mm_s": round(rop * 1e3, 4), "fault_trigger": fault, "brush_active": brush,
            "carousel_station": round(st["station"], 4),       # 0 = blank under the port, s = cup s under the port
        })
        st["t"] += dt

    # IDLE: 20 s health check (design)
    for _ in range(int(ctl["idle_health_check_s"])):
        log("IDLE")
    # ACTIVE: heater warm-up for the derived duration
    for _ in range(int(d["warmup"][env_id])):
        log("ACTIVE", heater=p_heat)

    lift_force = m_move * ph.g
    while st["hole"] < target - 1e-6:
        st["bite"] += 1
        start_index()
        bite_target = min(target, round(st["bite"] * bite, 6))
        bite_start = st["hole"]
        while st["hole"] < bite_target - 1e-6:
            # SEARCH: fast to hole_bottom - clearance, then slow until load-cell contact
            while st["z"] < st["hole"] - 1e-6:
                v = v_fast if st["z"] < st["hole"] - clear else v_search
                st["z"] = min(st["hole"], st["z"] + v * dt)
                contact = st["z"] >= st["hole"] - 1e-6
                log("SEARCH", wob=w_contact if contact else 0.0, p_feed_w=feed_power(lift_force, v))
            # DRILL
            fault = "NONE"
            while st["hole"] < bite_target - 1e-6:
                z = st["hole"]
                rpm = max(rpm_min, rpm_nom * float(np.clip((t_crit - st["T"]) / derate_band, derate_floor, 1.0)))
                pen = st["hole"] - bite_start
                a = ph.operating_point(z, pen, wob_set, rpm, False)
                b = ph.operating_point(z, pen, wob_set, rpm, True)
                op = b if (b["rop_m_s"] > 0 and (a["rop_m_s"] <= 0 or b["mse_pa"] < a["mse_pa"])) else a
                alt = a if op is b else b
                rop = op["rop_m_s"]
                dz = min(rop * dt, bite_target - st["hole"])
                st["hole"] += dz
                st["z"] = st["hole"]
                q_in = chi * op["p_face_w"]
                if op["torque_total_nm"] + t_brush_idle >= tau_fault:
                    fault = "TORQUE_OVERLOAD"
                elif st["T"] + (q_in / c_bit) * dt >= t_crit:
                    fault = "THERMAL_OVERLOAD"
                log("DRILL", wob=wob_set, rpm=rpm, torque=op["torque_total_nm"] + t_brush_idle,
                    perc=1 if op["percussion_active"] else 0,
                    p_drill=op["p_drill_elec_w"] + t_brush_idle * rpm * 2 * math.pi / 60 / ph.eta_rot,
                    p_feed_w=feed_power(wob_set, rop), mse=op["mse_pa"], rop=rop, fault=fault, q_in=q_in,
                    mode="rotary-percussive" if op["percussion_active"] else "rotary",
                    mse_alt=alt["mse_pa"] if alt["rop_m_s"] > 0 else float("inf"))
                if fault != "NONE":
                    break
            if fault == "THERMAL_OVERLOAD":
                # Report Sec. 4.3: stop, lift a few cm off bottom, wait for the bit to cool in the hole
                z_stand = max(0.0, st["hole"] - standoff)
                while st["z"] > z_stand + 1e-6:
                    st["z"] = max(z_stand, st["z"] - v_fast * dt)
                    log("RETRACTION", p_feed_w=feed_power(lift_force, v_fast), fault=fault)
                waited = 0
                while st["T"] > t_resume and waited < max_dwell:
                    log("RETRACTION", fault=fault)
                    waited += 1
                if st["T"] <= t_resume:
                    continue  # SEARCH back to bottom and resume the same bite
            # RETRACTION (bite complete, torque fault, or failed in-hole cooling): no rotation
            while st["z"] > z_sweep_start + 1e-6:
                st["z"] = max(z_sweep_start, st["z"] - v_fast * dt)
                log("RETRACTION", p_feed_w=feed_power(lift_force, v_fast), fault=fault,
                    brush=1 if st["z"] < -(brush_h - zone1) else 0)
            # TRANSFER: sweep the sampling flutes past the brush while rotating
            while st["z"] > z_sweep_end + 1e-6:
                st["z"] = max(z_sweep_end, st["z"] - v_sweep * dt)
                om = rpm_tr * 2 * math.pi / 60
                p_rot = t_brush * om / ph.eta_rot + ph.p_nl
                log("TRANSFER", rpm=rpm_tr, torque=t_brush, p_drill=p_rot, p_feed_w=feed_power(lift_force, v_sweep), brush=1)
            bite_start = st["hole"] if fault == "NONE" else bite_start
            if fault == "TORQUE_OVERLOAD":
                bite_start = st["hole"]  # flutes emptied -> continue the bite with fresh capacity

    # Final stow to home; the carousel turns the blank station under the port so every cup is sealed
    start_index()
    while st["z"] > z_home + 1e-6:
        st["z"] = max(z_home, st["z"] - v_fast * dt)
        log("IDLE", p_feed_w=feed_power(lift_force, v_fast))
    return pd.DataFrame(rec)


def abcd_from_telemetry(p: Dict[str, Any], d: Dict[str, Any], df: pd.DataFrame, env_id: str,
                        solar: Dict[str, Any]) -> Dict[str, Any]:
    dt = float(p["system"]["dt_s"])
    stor = p["power_system"]["energy_storage"]
    e_wh = float(df["power_w"].sum() * dt / 3600.0)
    t_h = float(len(df) * dt / 3600.0)
    drill = df[df["state"] == "DRILL"]
    bites = []
    for b, g in drill.groupby("bite_number"):
        allb = df[df["bite_number"] == b]
        bites.append({
            "bite_number": int(b),
            "z_range_m": [round(float(g["depth_m"].min()), 3), round(float(g["depth_m"].max()), 3)],
            "layer_name": str(g["layer_name"].iloc[-1]),
            "rop_mm_s": round(float(g["rop_mm_s"].mean()), 3),
            "drill_duration_s": int(len(g)),
            "cycle_duration_s": int(len(allb)),
            "mean_bite_power_w": round(float(g["power_w"].mean()), 1),
            "peak_torque_nm": round(float(g["torque_nm"].max()), 3),
            "mse_mj_m3": round(float(g["mse_mj_m3"].mean()), 2),
            "percussion_fraction": round(float(g["percussion_duty"].mean()), 2),
            "max_bit_temp_k": round(float(allb["bit_temp_k"].max()), 1),
            "faults": sorted(set(allb["fault_trigger"]) - {"NONE"}),
        })
    faults = df[(df["fault_trigger"] != "NONE") & (df["state"] == "DRILL")]
    cap = float(stor["nominal_capacity_wh"])
    min_bat = float(df["battery_wh"].min())
    chems = {}
    for key, ch in stor["chemistries"].items():
        eff = float(ch["round_trip_efficiency"])
        chems[key] = {
            "name": ch["name"],
            "specific_energy_wh_kg": ch["specific_energy_wh_kg"],
            "pack_mass_kg": round(cap / float(ch["specific_energy_wh_kg"]), 2),
            "round_trip_efficiency": eff,
            "solar_energy_to_refill_wh": round((cap - min_bat) / eff, 1),
            "min_operating_temp_k": ch["min_operating_temp_k"],
        }
    return {
        "environment_name": p["environments"][env_id]["name"],
        "A_target_depth_m": float(p["system"]["target_depth_m"]),
        "A_reached_depth_m": round(float(df["hole_bottom_m"].max()), 3),
        "A_bite_increment_m": float(p["system"]["bite_depth_m"]),
        "A_num_bites": int(df["bite_number"].max()),
        "B_total_operational_duration_hr": round(t_h, 3),
        "B_total_operational_duration_min": round(t_h * 60, 1),
        "B_pure_drilling_time_min": round(len(drill) * dt / 60, 2),
        "B_time_breakdown_min": {s: round(float((df["state"] == s).sum() * dt / 60), 2) for s in STATES},
        "C_wob_max_limit_n": d["wob"][env_id]["wob_limit_n"],
        "C_operational_wob_setpoint_n": d["wob"][env_id]["wob_setpoint_n"],
        "D_mean_electrical_power_w": round(e_wh / t_h, 1),
        "D_mean_drilling_power_w": round(float(drill["power_w"].mean()), 1),
        "D_peak_electrical_power_w": round(float(df["power_w"].max()), 1),
        "D_total_mission_energy_wh": round(e_wh, 1),
        "D_mean_solar_power_w": round(float(df["solar_w"].mean()), 1),
        "D_net_solar_power_margin_w": round(float(df["solar_w"].mean()) - e_wh / t_h, 1),
        "D_min_battery_wh": round(min_bat, 1),
        "D_max_depth_of_discharge_pct": round(100 * (1 - min_bat / cap), 2),
        "solar_day_mean_w": solar["configs"][solar["selected"]][env_id]["mean_w"],
        "num_faults": int(len(faults)),
        "fault_events": [{"time_s": float(r.time_s), "depth_m": float(r.depth_m), "bite": int(r.bite_number),
                          "fault": r.fault_trigger, "bit_temp_k": float(r.bit_temp_k), "torque_nm": float(r.torque_nm),
                          "layer": r.layer_name} for r in faults.itertuples()],
        "max_bit_temp_k": round(float(df["bit_temp_k"].max()), 1),
        "critical_bit_temp_k": float(d["critical_temp_k"][env_id]),
        "percussion_time_fraction": round(float(drill["percussion_duty"].mean()), 3),
        "storage_trade_study": chems,
        "bite_breakdown": bites,
    }


def build_keyframes(dfs: Dict[str, pd.DataFrame], d: Dict[str, Any], p: Dict[str, Any], stride: int = 4) -> Dict[str, Any]:
    aug, cup = d["auger"], d["brush"]["cup"]
    out: Dict[str, Any] = {"environments": {}, "sample": {
        # Bite-sample bookkeeping for the twin: flute fill length = penetration x A_hole B / A_flute
        "hole_area_m2": aug["hole_area_mm2"] / 1e6, "flute_area_m2": aug["flute_area_mm2"] / 1e6,
        "bulking_factor": float(p["cutting_model"]["cuttings_bulking_factor"]),
        "sampling_length_m": aug["sampling_length_mm"] / 1e3, "choking_onset_m": aug["choking_onset_m"],
        "cup_id_m": cup["id_m"], "cup_depth_m": cup["depth_m"], "cup_volume_m3": cup["volume_m3"],
    }}
    for env_id, df in dfs.items():
        change = df["state"] != df["state"].shift(1)
        fault = df["fault_trigger"] != "NONE"
        sub = df.loc[(df.index % stride == 0) | change | fault | (df.index == len(df) - 1)].reset_index(drop=True)
        frames = [{
            "idx": int(i), "t": float(r.time_s), "state": r.state, "bite": int(r.bite_number),
            "z_cm": float(r.depth_cm), "hole_cm": float(r.hole_bottom_cm), "layer": r.layer_name,
            "wob": float(r.wob_n), "wob_max": float(r.wob_max_n), "rpm": float(r.rpm), "torque": float(r.torque_nm),
            "perc": float(r.percussion_duty), "temp_k": float(r.bit_temp_k), "crit_k": float(r.crit_temp_k),
            "power_w": float(r.power_w), "solar_w": float(r.solar_w), "bat_wh": float(r.battery_wh),
            "sun_az": float(r.sun_az_deg), "sun_el": float(r.sun_el_deg),
            "mse": float(r.mse_mj_m3), "fault": r.fault_trigger, "brush": int(r.brush_active),
            "ucs": float(r.ucs_mpa), "rop": float(r.rop_mm_s), "station": float(r.carousel_station),
            # MSE of the mode the controller rejected; null = that mode cannot penetrate (or not drilling)
            "mse_alt": float(r.mse_alt_mj_m3) if r.mse_alt_mj_m3 > 0 else None,
        } for i, r in enumerate(sub.itertuples())]

        def first(mask) -> Optional[int]:
            idx = sub.index[mask].tolist()
            return int(idx[0]) if idx else None

        drill3 = sub.index[(sub["state"] == "DRILL") & (sub["bite_number"] == 3)].tolist()
        out["environments"][env_id] = {
            "total_frames": len(frames),
            "wob_setpoint_n": float(d["wob"][env_id]["wob_setpoint_n"]),
            "bookmarks": {
                "SEARCH": first((sub["state"] == "SEARCH") & (sub["bite_number"] == 3)) or 0,
                "DRILL_BITE_3": int(drill3[len(drill3) // 2]) if drill3 else 0,
                "FAULT_RETRACT": first(sub["fault_trigger"] != "NONE"),
                "TRANSFER": first((sub["state"] == "TRANSFER") & (sub["bite_number"] == 3)) or 0,
            },
            "strata": d["strata"][env_id],
            "frames": frames,
        }
    return out


def main() -> None:
    EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = load_system_params()
    d = derive_all(p)
    solar = trade_study(p)
    cfg = solar["selected"]
    dfs = {env_id: simulate(p, d, env_id, cfg) for env_id in p["environments"]}
    for env_id, df in dfs.items():
        df.to_csv(EXPORTS_DATA_DIR / f"telemetry_{env_id}.csv", index=False)
    kf = build_keyframes(dfs, d, p)
    for path in (EXPORTS_DATA_DIR / "kinematic_keyframes.json", DOCS_ASSETS_DATA_DIR / "kinematic_keyframes.json"):
        path.write_text(json.dumps(kf, separators=(",", ":")))
    abcd = {
        "system_name": p["system"]["name"],
        "lander_mass_kg": p["system"]["lander_mass_kg"],
        "solar_configuration": CONFIG_LABELS[cfg],
        "environments": {e: abcd_from_telemetry(p, d, dfs[e], e, solar) for e in dfs},
    }
    # Off-nominal case: bite length longer than the derived sampling-flute capacity (z_crit).
    # Demonstrates the torque-fault recovery that the physics predicts (nothing is scripted).
    import copy
    p_off = copy.deepcopy(p)
    p_off["system"]["bite_depth_m"] = 0.15
    abcd["off_nominal"] = {"description": "Bite length 15 cm > choking onset z_crit = %.3f m" % d["auger"]["choking_onset_m"]}
    for env_id in p["environments"]:
        df_off = simulate(p_off, d, env_id, cfg)
        df_off.to_csv(EXPORTS_DATA_DIR / f"telemetry_{env_id}_offnominal.csv", index=False)
        abcd["off_nominal"][env_id] = abcd_from_telemetry(p_off, d, df_off, env_id, solar)
    # One-at-a-time sensitivity of the A-B-C-D results to the least certain coefficients
    sens = []
    for key, vals in (("percussive_se_to_ucs", (1.0, 3.0, 5.0)),
                      ("heat_to_bit_fraction", (0.15, 0.25, 0.35)),
                      ("cuttings_bulking_factor", (1.3, 1.5, 1.6))):
        for v in vals:
            p_s = copy.deepcopy(p)
            p_s["cutting_model"][key] = v
            d_s = derive_all(p_s)
            row = {"parameter": key, "value": v}
            for env_id in p["environments"]:
                a_s = abcd_from_telemetry(p_s, d_s, simulate(p_s, d_s, env_id, cfg), env_id, solar)
                row[env_id] = {"duration_h": a_s["B_total_operational_duration_hr"],
                               "drilling_min": a_s["B_pure_drilling_time_min"],
                               "mean_drilling_w": a_s["D_mean_drilling_power_w"],
                               "faults": a_s["num_faults"], "max_bit_temp_k": a_s["max_bit_temp_k"]}
            sens.append(row)
    abcd["sensitivity"] = sens
    (EXPORTS_DATA_DIR / "abcd_summary.json").write_text(json.dumps(abcd, indent=2))
    print("=== Module 2: 6-state mission simulation ===")
    for e, a in abcd["environments"].items():
        print(f"  {e}: A={a['A_reached_depth_m']} m in B={a['B_total_operational_duration_hr']:.2f} h "
              f"(drilling {a['B_pure_drilling_time_min']:.1f} min) | C={a['C_operational_wob_setpoint_n']} N "
              f"| D mean {a['D_mean_electrical_power_w']} W / drilling {a['D_mean_drilling_power_w']} W / peak "
              f"{a['D_peak_electrical_power_w']} W | solar {a['D_mean_solar_power_w']} W | DoD {a['D_max_depth_of_discharge_pct']}% "
              f"| faults {a['num_faults']} | Tmax {a['max_bit_temp_k']}/{a['critical_bit_temp_k']} K")
        print(f"     time (min): {a['B_time_breakdown_min']}")
        o = abcd["off_nominal"][e]
        print(f"     off-nominal 15 cm bites: {o['B_total_operational_duration_hr']:.2f} h, faults {o['num_faults']} "
              f"{sorted(set(f['fault'] for f in o['fault_events']))}")


if __name__ == "__main__":
    main()
