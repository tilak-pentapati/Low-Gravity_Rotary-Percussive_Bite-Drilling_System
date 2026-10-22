"""
Passive brushing-station derivations: worm-wheel mesh of the bristle tufts with the sampling flights,
bristle wire and interference, contact force budget, brush drag torque, spring-loaded pivot arm,
axle bushings, sample chute and cup, and the stowed (bake-out / launch) position. Called from
src/design.py; every result is written to the derivation ledger.

Mesh (worm and wheel on crossed axes)
  The auger flight is a single-start worm of lead p; each bristle tuft is a wheel tooth that sits in the flute.
  Wheel circular pitch = p at the pitch radius R_p = N p / (2 pi), placed on the mean flute radius
  (a = R_p + r_m). The tufts lean at the flight lead angle, as a worm wheel's helical teeth do.
  N = smallest tuft count whose tuft holes (thickness s_t = w - 2 j) leave the hub ligament at the hub
  radius and whose tufts enter and leave the flute without touching a flight face:
      |rho sin(theta) - R_p theta| + s_t/2 cos(theta) <= w/2   for every tuft point inside the flute.
  Ratio: the brush turns one tuft per auger revolution, omega_b = omega_a p / (2 pi R_p) = omega_a / N.

Bristles (steel filament as a cantilever, Stango & Shia 1987 brush model)
  F = 3 E I delta / L^3,  sigma = 32 F L / (pi d^3) = 3 E d delta / (2 L^2)
  fatigue 0 -> sigma each brush revolution, Goodman with Zimmerli endurance converted to bending
  (S = S_s / 0.577, Shigley Sec. 10-12).
  Strip requirement: tip pressure p = (3/16) phi E d^2 delta / L^3 must exceed the Rankine passive
  pressure of the cuttings, sigma_p = 2 c sqrt(K_p) + gamma h K_p, K_p = tan^2(45 + phi/2).
  Force budget: the radial contact force pushes the string sideways; with the bit above the guide
  bushing (end of TRANSFER, re-entry) the bit tip may move at most the bushing lead-in chamfer.
  Choice: the thickest stock wire (longest tip-wear life) that meets fatigue and the force budget at the
  minimum interference, and the strip requirement.

Drag torque on the auger (flutes full, conservative)
  T = F_plough r_m + mu N r_root + T_bushings / N_tufts,   F_plough = w (gamma h^2 K_p / 2 + 2 c h sqrt(K_p))
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


def _mesh_clearance(n: int, p: float, r_od: float, r_root: float, w: float, s_t: float, delta: float,
                    c_hub: float) -> Dict[str, float]:
    r_m = 0.5 * (r_od + r_root)
    r_p = n * p / (2 * math.pi)
    a = r_p + r_m
    r_t = a - r_root + delta
    r_h = a - r_od - c_hub
    worst = -1.0
    for th in np.linspace(0.0, 1.4, 700):
        for rho in np.linspace(r_h, r_t, 30):
            if a - rho * math.cos(th) > r_od:
                continue
            worst = max(worst, abs(rho * math.sin(th) - r_p * th) + 0.5 * s_t * math.cos(th) - 0.5 * w)
    return {"n": n, "r_p": r_p, "a": a, "r_t": r_t, "r_h": r_h, "interference": worst}


def _root_contact(n_grid: int, r_root: float, a: float, r_t: float, s_t: float, b: float) -> List[tuple]:
    """Bristle tip deflections against the flute root over the tuft footprint at the line of centres."""
    out = []
    for x in np.linspace(-b / 2, b / 2, n_grid):
        if abs(x) >= r_root:
            continue
        y_root = math.sqrt(r_root ** 2 - x ** 2)
        for z in np.linspace(-s_t / 2, s_t / 2, n_grid):
            d = y_root - (a - math.sqrt(r_t ** 2 - z ** 2))
            if d > 0:
                out.append((x, z, d))
    return out


def derive_brush(p: Dict[str, Any], led, auger: Dict[str, Any], string: Dict[str, Any]) -> Dict[str, Any]:
    a_g = p["auger_geometry"]
    bs = p["actuators"]["brush_station"]
    cm = p["cutting_model"]
    cr = p["criteria"]
    wire = p["materials"]["hammer_spring_wire"]
    e_w = float(wire["youngs_modulus_pa"])
    r_od = float(a_g["outer_diameter_m"]) / 2
    r_gauge = float(a_g["gauge_diameter_m"]) / 2
    r_root = auger["root_od_mm"] / 2e3
    p_s = auger["sampling_pitch_mm"] / 1e3
    t_f = float(a_g["sampling_flight_thickness_m"])
    h_f = r_od - r_root
    r_m = 0.5 * (r_od + r_root)
    w = p_s - t_f
    j = float(bs["tuft_backlash_m"])
    s_t = w - 2 * j
    delta = float(bs["min_interference_m"])
    c_hub = float(bs["hub_flight_clearance_m"])
    lig = float(bs["hub_ligament_m"])
    out: Dict[str, Any] = {}

    # ---- Worm-wheel mesh ----------------------------------------------------------------------------
    rows = []
    mesh = None
    for n in range(6, 60):
        m = _mesh_clearance(n, p_s, r_od, r_root, w, s_t, delta, c_hub)
        m["hub_fit"] = 2 * math.pi * m["r_h"] - n * (s_t + lig)
        rows.append({k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()})
        if m["interference"] <= 0 and m["hub_fit"] >= 0:
            mesh = m
            break
    assert mesh is not None
    n_t, r_p, a_ax, r_t, r_h = mesh["n"], mesh["r_p"], mesh["a"], mesh["r_t"], mesh["r_h"]
    lead_deg = math.degrees(math.atan(p_s / (2 * math.pi * r_m)))
    led.add("brush_tuft_count", n_t, "-", "min N: tufts clear the flights (worm-wheel entry) and fit the hub",
            f"tuft {s_t*1e3:.1f} mm thick in an {w*1e3:.1f} mm flute (backlash {j*1e3:.1f} mm each side), {lig*1e3:.0f} mm hub ligament")
    led.add("brush_pitch_radius_mm", round(r_p * 1e3, 2), "mm", "R_p = N p / (2 pi)", f"wheel circular pitch = flight pitch {p_s*1e3:.1f} mm")
    led.add("brush_axle_offset_mm", round(a_ax * 1e3, 2), "mm", "a = R_p + r_m", "auger axis to brush axle (pitch circle on the mean flute radius)")
    led.add("brush_tip_radius_mm", round(r_t * 1e3, 2), "mm", "R_t = a - r_root + delta", f"tips pass {delta*1e3:.1f} mm beyond the flute root")
    led.add("brush_tuft_lean_deg", round(lead_deg, 2), "deg", "lambda = atan(p / (2 pi r_m))", "tufts lean at the flight lead angle (worm-wheel helix)")
    led.add("brush_speed_ratio", n_t, "-", "omega_a / omega_b = N (single-start worm)",
            f"{float(bs['transfer_rpm'])/n_t:.2f} rpm at the {bs['transfer_rpm']:.0f} rpm transfer speed; "
            f"transport pitch / sampling pitch = {auger['transport_pitch_mm']/auger['sampling_pitch_mm']:.2f}")
    out["mesh"] = {"tufts": n_t, "r_p_m": r_p, "axle_offset_m": a_ax, "r_tip_m": r_t, "r_hub_m": r_h, "lean_deg": lead_deg,
                   "tuft_thickness_m": s_t, "rows": rows}

    # ---- Cuttings strength (Rankine passive) -----------------------------------------------------------
    phi_c = math.radians(float(bs["cuttings_friction_angle_deg"]))
    k_p = math.tan(math.pi / 4 + phi_c / 2) ** 2
    c_r = float(cm["remoulded_cuttings_cohesion_kpa"]) * 1e3
    g_max = max(float(e["gravity_m_s2"]) for e in p["environments"].values())
    rho_max = max(float(L["density_kg_m3"]) for e in p["environments"].values() for L in e["stratigraphy"])
    gam = rho_max / float(cm["cuttings_bulking_factor"]) * g_max
    sig_p = 2 * c_r * math.sqrt(k_p) + gam * h_f * k_p
    f_plough = w * (0.5 * gam * h_f ** 2 * k_p + 2 * c_r * h_f * math.sqrt(k_p))
    led.add("cuttings_passive_pressure_kpa", round(sig_p / 1e3, 2), "kPa", "sigma_p = 2 c sqrt(K_p) + gamma h K_p",
            f"Rankine passive, K_p = {k_p:.2f}; c = {c_r/1e3:.1f} kPa, Mars gravity, densest layer bulked")

    # ---- Bristle wire choice --------------------------------------------------------------------------
    l_b = h_f + delta + c_hub
    face_b = 2 * math.sqrt(r_od ** 2 - (r_root - delta) ** 2)
    phi_pack = float(bs["tuft_packing_fraction"])
    mu_b = float(bs["bristle_friction_coeff"])
    s_su = 0.67 * float(wire["ultimate_tensile_pa"])
    s_se = float(wire["zimmerli_ssa_pa"]) / (1.0 - float(wire["zimmerli_ssm_pa"]) / s_su)
    s_e, s_ut = s_se / 0.577, float(wire["ultimate_tensile_pa"])
    k_lat = 1.0 / string["beam"]["tip_compliance_m_per_n"]
    f_budget = k_lat * string["bushing"]["leadin_m"]
    contact = _root_contact(61, r_root, a_ax, r_t, s_t, face_b)
    cell = (face_b / 60) * (s_t / 60)
    sweep = []
    chosen = None
    for d in sorted(float(x) for x in bs["bristle_wire_options_m"]):
        inertia = math.pi * d ** 4 / 64
        dens = phi_pack / (math.pi * d ** 2 / 4)
        n_force = sum(3 * e_w * inertia * dd / l_b ** 3 for _, _, dd in contact) * dens * cell
        sig = 1.5 * e_w * d * delta / l_b ** 2
        n_f = 1.0 / (0.5 * sig / s_e + 0.5 * sig / s_ut)
        p_tip = 3 / 16 * phi_pack * e_w * d ** 2 * delta / l_b ** 3
        ok = n_f >= float(cr["structural_fos_fatigue"]) and n_force <= f_budget and p_tip >= sig_p
        row = {"wire_mm": round(d * 1e3, 3), "contact_force_n": round(n_force, 3), "stress_mpa": round(sig / 1e6, 0),
               "fos_fatigue": round(n_f, 2), "tip_pressure_kpa": round(p_tip / 1e3, 1), "ok": ok}
        sweep.append(row)
        if ok:
            chosen = dict(row, d=d, n_force=n_force, p_tip=p_tip, bristles=int(dens * s_t * face_b))
    assert chosen is not None, "no stock bristle wire meets the brush criteria"
    d_w, n_force = chosen["d"], chosen["n_force"]
    led.add("brush_contact_force_budget_n", round(f_budget, 3), "N", "F <= k_string x lead-in chamfer",
            f"string stiffness at the brush {k_lat:.0f} N/m (bit tip) with the bit above the guide bushing")
    led.add("bristle_wire_mm", chosen["wire_mm"], "mm", "thickest stock wire: Goodman, force budget, strip pressure",
            f"17-7PH, free length {l_b*1e3:.1f} mm, {chosen['bristles']} bristles per tuft ({phi_pack:.0%} fill)")
    led.add("bristle_stress_mpa", chosen["stress_mpa"], "MPa", "sigma = 3 E d delta / (2 L^2)",
            f"Goodman FOS {chosen['fos_fatigue']:.2f} (0 -> sigma once per brush turn)")
    led.add("bristle_tip_pressure_kpa", chosen["tip_pressure_kpa"], "kPa", "p = (3/16) phi E d^2 delta / L^3",
            f"{chosen['p_tip']/sig_p:.0f} x the cuttings' passive pressure: the root is scraped clean")
    led.add("brush_contact_force_n", round(n_force, 3), "N", "N = sum 3 E I delta_i / L^3 over the root contact",
            f"{n_force/f_budget*100:.0f} % of the force budget")
    out["bristle"] = {"wire_m": d_w, "free_len_m": l_b, "face_m": face_b, "sweep": sweep, "contact_n": n_force,
                      "stress_mpa": chosen["stress_mpa"], "fos": chosen["fos_fatigue"], "per_tuft": chosen["bristles"],
                      "budget_n": f_budget, "tip_pressure_pa": chosen["p_tip"], "passive_pa": sig_p}

    # ---- Sweep speed for the TRANSFER state -----------------------------------------------------------
    patch = 2 * math.sqrt(r_t ** 2 - (a_ax - r_root) ** 2)
    n_tr = float(bs["transfer_rpm"])
    v_sweep = led.add("brush_sweep_speed_mm_s", round(patch / float(cr["brush_passes_per_flute"]) * n_tr / 60 * 1e3, 2), "mm/s",
                      "v = (2 sqrt(R_t^2 - (a - r_root)^2) / N_pass) x n/60", f"root-contact patch {patch*1e3:.1f} mm along the auger")
    t_tr = led.add("transfer_duration_s", math.ceil(auger["sampling_length_mm"] / v_sweep), "s", "t = L_s / v_sweep")
    out["sweep"] = {"patch_m": patch, "v_mm_s": v_sweep, "duration_s": t_tr}

    # ---- Pivot arm, torsion spring, bushings ----------------------------------------------------------
    l_arm = float(bs["arm_length_m"])
    m_spring = n_force * l_arm
    pin = float(bs["pivot_pin_d_m"])
    clr = float(bs["spring_coil_clearance_m"])
    n_a = math.ceil(float(cr["spring_min_active_coils"]))
    s_bend = float(wire["allowable_shear_pa"]) / 0.577
    spring = None
    for d in sorted(float(x) for x in bs["spring_wire_options_m"]):
        dm = pin + clr + d
        c = dm / d
        k_i = (4 * c * c - c - 1) / (4 * c * (c - 1))
        sig = k_i * 32 * m_spring / (math.pi * d ** 3)
        if 4.0 <= c <= 12.0 and sig * float(cr["structural_fos_yield"]) <= s_bend:
            k_t = e_w * d ** 4 / (64 * dm * n_a)
            spring = {"wire_m": d, "coil_d_m": dm, "index": c, "stress_mpa": sig / 1e6, "rate_nm_rad": k_t,
                      "preload_deg": math.degrees(m_spring / k_t), "active_coils": n_a, "moment_nm": m_spring}
            break
    assert spring is not None
    step = r_gauge - r_od                                    # bit crown passes: tips pushed out by the gauge step
    d_force = spring["rate_nm_rad"] * (step / l_arm) / l_arm
    led.add("brush_spring_preload_n", round(n_force, 3), "N", "F_pre = N (arm floats on the bristles)",
            "preload equals the required bristle force; the arm follows the string as it deflects")
    led.add("brush_spring_wire_mm", round(spring["wire_m"] * 1e3, 2), "mm", "thinnest stock wire: K_i 32 M / (pi d^3) x FOS <= S_sy/0.577",
            f"torsion spring on the {pin*1e3:.0f} mm pivot pin: index {spring['index']:.1f}, {n_a} coils, "
            f"wound up {spring['preload_deg']:.0f} deg, {spring['stress_mpa']:.0f} MPa")
    led.add("brush_force_rise_crown_pct", round(100 * d_force / n_force, 2), "%", "dF/F = k_t (step / L_arm) / (L_arm F)",
            f"when the {step*1e3:.1f} mm gauge step of the bit crown lifts the brush")
    bush = bs["bushing"]
    ax_d = float(bs["axle_d_m"])
    om_b = float(bs["transfer_rpm"]) / n_t * 2 * math.pi / 60 + v_sweep / 1e3 / r_p
    load = 0.5 * n_force
    pv = load / (ax_d * float(bush["length_m"])) * om_b * ax_d / 2
    led.add("brush_bushing_pv_pa_m_s", round(pv, 2), "Pa m/s", "PV = (F/2)/(d L) x omega_b d/2",
            f"{bush['name']}: {pv/float(bush['pv_limit_pa_m_s'])*100:.4f} % of the {float(bush['pv_limit_pa_m_s'])/1e6:.2f} MPa m/s limit")
    t_bush = float(bush["friction_coeff"]) * n_force * ax_d / 2

    # ---- Drag torque on the auger ---------------------------------------------------------------------
    t_plough = f_plough * r_m
    t_fric = mu_b * n_force * r_root
    t_worm = t_bush / n_t
    t_drag = t_plough + t_fric + t_worm
    led.add("brush_drag_torque_nm", round(t_drag, 5), "N m", "T = F_plough r_m + mu N r_root + T_bush / N",
            f"plough {t_plough*1e3:.2f} + bristle friction {t_fric*1e3:.2f} + brush bushings {t_worm*1e3:.3f} mN m (full flutes)")
    t_idle = t_fric + t_worm
    led.add("brush_idle_drag_torque_nm", round(t_idle, 5), "N m", "T = mu N r_root + T_bush / N",
            "empty flights passing the brush while drilling")
    out["drag"] = {"total_nm": t_drag, "idle_nm": t_idle, "plough_nm": t_plough, "friction_nm": t_fric, "bushing_nm": t_worm}

    # ---- Chute and sample cup -------------------------------------------------------------------------
    bulk = float(cm["cuttings_bulking_factor"])
    v_bite = float(p["system"]["bite_depth_m"]) * auger["hole_area_mm2"] / 1e6 * bulk
    v_cup = v_bite * float(cr["cuttings_retention_margin"])
    led.add("bite_cuttings_volume_cm3", round(v_bite * 1e6, 1), "cm^3", "V = z_bite A_hole B")
    mu_c = float(bs["cuttings_bronze_friction_coeff"])
    ang = math.ceil(math.degrees(math.atan(mu_c)) + float(bs["chute_angle_margin_deg"]))
    led.add("chute_angle_deg", ang, "deg", "theta >= atan(mu_w) + margin",
            f"wall friction angle {math.degrees(math.atan(mu_c)):.1f} deg on bronze (g-independent for dry flow)")
    lay = p["cad_layout"]
    s_str = p["auger_string"]
    wall = float(bs["chute_wall_m"])
    gap = float(bs["chute_side_clearance_m"])
    width = face_b + 2 * gap
    # Inlet lip wraps the auger at the largest excursion of the bit re-entering the bushing
    y_in = r_gauge + float(s_str["guide_bushing_radial_clearance_m"]) + float(s_str["guide_bushing_leadin_m"])
    z_in = float(lay["brush_axle_height_m"]) - r_t - gap
    tan_c = math.tan(math.radians(ang))
    # Cups hang under the mast foot plate on the carousel turntable; the plate is their cover. Depth is set by
    # the ground clearance (deeper cups -> smaller bore -> smaller carousel); the port cup's flange clears the
    # guide-bushing flange under the plate, and the chute outlet sits over the port.
    ca = p["actuators"]["sample_carousel"]
    flange = float(ca["seal_flange_width_m"])
    z_plate0 = float(lay["foot_top_m"]) - float(p["drill_mast"]["foot_plate_thickness_m"])
    z_rim = z_plate0 - float(ca["seal_gap_m"])
    z_tt0 = float(ca["ground_clearance_m"])
    h_cup = z_rim - wall - float(ca["turntable_thickness_m"]) - z_tt0
    d_cup = max(width + 2 * wall, math.sqrt(4 * v_cup / (math.pi * h_cup)))
    r_bflange = string["bushing"]["bore_m"] / 2 + 0.009          # guide-bushing flange under the plate (src/cad_auger.py)
    y_cup = max(y_in + 0.5 * d_cup, r_bflange + float(ca["radial_clearance_m"]) + 0.5 * d_cup + flange)
    z_out = z_in - (y_cup - y_in) * tan_c
    assert z_out >= float(lay["foot_top_m"]) + gap, "chute outlet below the foot-plate port"
    led.add("sample_cup_id_mm", round(d_cup * 1e3, 1), "mm", "pi/4 D^2 h >= V_bite x margin",
            f"{v_cup*1e6:.1f} cm^3 ({cr['cuttings_retention_margin']:.0%} of one bite); {h_cup*1e3:.1f} mm deep, the most the "
            f"{float(ca['ground_clearance_m'])*1e3:.0f} mm ground clearance under the foot plate allows")
    # Flowing-layer depth: brush delivery rate over the slowest sliding flow (Moon), at mid-chute
    g_min = min(float(e["gravity_m_s2"]) for e in p["environments"].values())
    run = (y_cup - y_in) / math.cos(math.radians(ang))
    acc_c = g_min * (math.sin(math.radians(ang)) - mu_c * math.cos(math.radians(ang)))
    q_dot = v_bite / t_tr
    h_flow = q_dot / (width * math.sqrt(acc_c * run))
    h_wall = float(bs["chute_wall_height_m"])
    led.add("chute_flow_depth_mm", round(h_flow * 1e3, 3), "mm", "h = Q / (w sqrt(2 g' L/2)),  g' = g (sin th - mu cos th)",
            f"brush delivers {q_dot*1e6:.2f} cm^3/s; lunar sliding flow, {h_wall*1e3:.0f} mm side walls; run {run*1e3:.0f} mm")
    out["chute"] = {"angle_deg": ang, "width_m": width, "y_in_m": y_in, "z_in_m": z_in, "y_out_m": y_cup, "z_out_m": z_out,
                    "wall_h_m": h_wall, "wall_m": wall, "flow_depth_m": h_flow}
    out["cup"] = {"id_m": d_cup, "depth_m": h_cup, "y_m": y_cup, "z0_m": z_rim - h_cup - wall, "rim_m": z_rim,
                  "volume_m3": v_cup, "wall_m": wall, "flange_m": flange}
    out["carousel"] = _carousel(p, led, out["cup"], r_bflange, v_bite)

    # ---- Stow / bake-out position ---------------------------------------------------------------------
    dy = r_t - (a_ax - r_gauge) + float(bs["stow_clearance_m"])
    stow = math.degrees(math.asin(dy / l_arm))
    led.add("brush_stow_angle_deg", round(stow, 1), "deg", "psi = asin((R_t - (a - r_gauge) + c) / L_arm)",
            f"arm swung away so the tips clear the bit gauge by {float(bs['stow_clearance_m'])*1e3:.0f} mm: launch lock + dry-heat bake-out position")
    out["arm"] = {"length_m": l_arm, "pivot_pin_m": pin, "stow_deg": stow, "spring": spring, "axle_d_m": ax_d,
                  "bushing_len_m": float(bush["length_m"])}
    return out


def _carousel(p: Dict[str, Any], led, cup: Dict[str, Any], r_bflange: float, v_bite: float) -> Dict[str, Any]:
    """Indexing cup carousel under the mast foot plate: one sealed cup per bite plus blank station(s).

    Pitch circle  R = N_st (D_f + gap) / (2 pi),  D_f = cup bore + 2 x seal flange (cups just fit round it)
    Centre        a = y_cup + R from the auger axis, so the port cup sits under the chute outlet
    Index move    cycloidal over t: alpha_max = 2 pi theta / t^2, omega_max = 2 theta / t,  theta = 2 pi / N_st
    Torque        T = k_s (T_seal + T_hub) + k_i J alpha_max          (ECSS-E-ST-33-01C motorisation factors)
                  T_seal = mu q (N pi D_s) R  (each cup's face seal drags on the plate),  T_hub = mu_b m g r_hub
    """
    ca = p["actuators"]["sample_carousel"]
    cr = p["criteria"]
    mats = p["materials"]
    n_cups = math.ceil(round(float(p["system"]["target_depth_m"]) / float(p["system"]["bite_depth_m"]), 6))
    n_st = n_cups + int(ca["blank_stations"])
    d_f = cup["id_m"] + 2 * cup["flange_m"]
    r_pc = n_st * (d_f + float(ca["cup_gap_m"])) / (2 * math.pi)
    a_c = cup["y_m"] + r_pc
    c_rad = float(ca["radial_clearance_m"])
    r_tt = r_pc + d_f / 2
    near = cup["y_m"] - d_f / 2
    assert near >= r_bflange + c_rad - 1e-9, "cups would hit the guide-bushing flange"
    led.add("carousel_stations", n_st, "-", "N = ceil(depth / bite) cups + blank station",
            f"{n_cups} sealed cups, one per bite; the blank station sits under the port at launch and after the last bite")
    led.add("carousel_pitch_radius_mm", round(r_pc * 1e3, 1), "mm", "R = N_st (D_f + gap) / (2 pi)",
            f"seal flange {d_f*1e3:.1f} mm, {float(ca['cup_gap_m'])*1e3:.0f} mm between neighbours; axis {a_c*1e3:.0f} mm "
            f"fore of the auger, port-cup flange {(near - r_bflange)*1e3:.1f} mm from the guide-bushing flange")

    # Masses (6061-T6 turntable and cups, motor from its envelope), with growth allowance
    rho_al = float(mats["mast_aluminium"]["density_kg_m3"])
    mga = 1 + float(p["drill_mast"]["mass_growth_allowance"])
    t_tt = float(ca["turntable_thickness_m"])
    w = cup["wall_m"]
    d_o = cup["id_m"] + 2 * w
    m_cup = rho_al * (math.pi * d_o * (cup["depth_m"] + w) * w + math.pi / 4 * d_o ** 2 * w
                      + math.pi / 4 * (d_f ** 2 - d_o ** 2) * 0.001)
    m_tt = rho_al * t_tt * math.pi * r_tt ** 2
    mo = ca["index_motor"]
    m_mot = float(p["drill_mast"]["actuator_mean_density_kg_m3"]) * math.pi / 4 * mo["od_m"] ** 2 * mo["length_m"]
    m_rot = (m_tt + n_cups * m_cup) * mga
    m_tot = m_rot + m_mot * mga
    led.add("carousel_mass_kg", round(m_tot, 3), "kg", "sum rho V (1 + MGA)",
            f"turntable {m_tt*1e3:.0f} g, {n_cups} cups x {m_cup*1e3:.0f} g, index motor {m_mot*1e3:.0f} g; "
            f"hung from the mast foot plate, which grows a disc of radius {(r_tt + c_rad)*1e3:.0f} mm to cover the cups")

    # Worst case for the drive: every cup full of the densest layer's cuttings, Mars gravity
    rho_max = max(float(L["density_kg_m3"]) for e in p["environments"].values() for L in e["stratigraphy"])
    m_smp = v_bite * rho_max / float(p["cutting_model"]["cuttings_bulking_factor"])
    j = 0.5 * m_tt * mga * r_tt ** 2 + n_cups * (m_cup * mga + m_smp) * r_pc ** 2
    g_max = max(float(e["gravity_m_s2"]) for e in p["environments"].values())
    d_s = 0.5 * (cup["id_m"] + d_f)
    t_seal = float(ca["seal_friction_coeff"]) * float(ca["seal_line_load_n_m"]) * n_cups * math.pi * d_s * r_pc
    t_hub = float(ca["hub_bearing_friction_coeff"]) * (m_rot + n_cups * m_smp) * g_max * mo["od_m"] / 2
    th = 2 * math.pi / n_st
    t_ix = float(ca["index_time_s"])
    alpha = 2 * math.pi * th / t_ix ** 2
    omega = 2 * th / t_ix
    t_req = float(cr["motorisation_static_factor"]) * (t_seal + t_hub) + float(cr["motorisation_inertial_factor"]) * j * alpha
    assert t_req <= float(mo["rated_torque_nm"]), "carousel index motor too small"
    p_ix = (t_seal + t_hub + j * alpha) * omega / float(mo["efficiency"])
    led.add("carousel_index_torque_nm", round(t_req, 4), "N m", "T = 2 (T_seal + T_hub) + 1.25 J alpha",
            f"face seals {t_seal*1e3:.0f} mN m + hub bearing {t_hub*1e3:.3f} mN m + inertia {j*alpha*1e3:.3f} mN m "
            f"(J = {j*1e4:.1f} kg cm^2, all cups full); motor rated {float(mo['rated_torque_nm']):.2f} N m")
    led.add("carousel_index_power_w", round(p_ix, 3), "W", "P = T omega_max / eta",
            f"{math.degrees(th):.1f} deg in {t_ix:.0f} s while the auger descends, {p_ix * t_ix:.2f} J per index; "
            "empty and balanced at launch, so linear launch loads put no torque on the drive")
    return {"cups": n_cups, "stations": n_st, "pitch_radius_m": r_pc, "axis_offset_m": a_c, "turntable_r_m": r_tt,
            "plate_disc_r_m": r_tt + c_rad, "flange_d_m": d_f, "turntable_t_m": t_tt, "turntable_z0_m": float(ca["ground_clearance_m"]),
            "seal_gap_m": float(ca["seal_gap_m"]), "mass_kg": m_tot, "rotating_mass_kg": m_rot, "inertia_kg_m2": j,
            "index_time_s": t_ix, "index_torque_nm": t_req, "index_power_w": p_ix, "motor": dict(mo), "sample_mass_kg": m_smp,
            "seal_torque_nm": t_seal}
