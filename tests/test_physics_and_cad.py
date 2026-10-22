"""
Verification suite: checks physical laws, derivation consistency and exports.

The tests assert relationships that must hold if the physics is right
(e.g. WOB set-point = m g / FOS, rotary threshold > Lunar WOB in icy rock, choking only
beyond the derived flute capacity, energy conservation of the battery log) rather than
hard-coded target numbers.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import trimesh

from src.design import derive_all, ice_vapour_pressure_pa, load_system_params, mohr_coulomb_ucs_pa
from src.solar import trade_study
from src.terramechanics import PROJECT_ROOT, Physics

P = load_system_params()
D = derive_all(P)
DATA = PROJECT_ROOT / "exports" / "data"


def test_wob_is_lift_off_over_fos() -> None:
    m = P["system"]["lander_mass_kg"]
    fos = P["criteria"]["wob_lift_off_fos"]
    for env_id, env in P["environments"].items():
        w = D["wob"][env_id]
        assert w["wob_limit_n"] == pytest.approx(m * env["gravity_m_s2"], abs=0.01)
        assert w["wob_setpoint_n"] == pytest.approx(m * env["gravity_m_s2"] / fos, abs=0.01)


def test_percussion_needs_less_wob_than_available() -> None:
    """The hammer must stay seated: minimum WOB for percussion below both set-points."""
    for env_id in P["environments"]:
        assert D["hammer"]["min_wob_n"] < D["wob"][env_id]["wob_setpoint_n"]
    assert D["hammer"]["natural_hz"] > 2 * D["hammer"]["blow_frequency_hz"]
    assert D["hammer"]["spring_shear_mpa"] * 1e6 < P["materials"]["hammer_spring_wire"]["allowable_shear_pa"]


def test_auger_root_meets_fatigue_and_flute_holds_one_bite() -> None:
    a = D["auger"]
    chosen = [r for r in a["root_sweep"] if r["root_od_mm"] == a["root_od_mm"]][0]
    assert chosen["fos_fatigue"] >= P["criteria"]["structural_fos_fatigue"]
    assert a["choking_onset_m"] >= P["system"]["bite_depth_m"] * P["criteria"]["cuttings_retention_margin"] - 1e-3
    # Retention: helix angle at mean radius below the friction angle
    d_mean = 0.5 * (P["auger_geometry"]["outer_diameter_m"] * 1e3 + a["root_od_mm"])
    assert math.degrees(math.atan(a["sampling_pitch_mm"] / (math.pi * d_mean))) <= math.degrees(
        math.atan(P["auger_geometry"]["wall_friction_coeff_auger"])) + 1e-6


def test_rotary_stalls_in_lunar_ice_but_percussion_penetrates() -> None:
    ph = Physics(P, D, "moon")
    wob = D["wob"]["moon"]["wob_setpoint_n"]
    rot = ph.cutting(0.9, wob, 150.0, False)
    rp = ph.cutting(0.9, wob, 150.0, True)
    assert rot["w_threshold_n"] > D["wob"]["moon"]["wob_limit_n"]
    assert rot["rop_m_s"] == 0.0
    assert rp["rop_m_s"] > 0.0


def test_choking_only_beyond_derived_capacity() -> None:
    ph = Physics(P, D, "mars")
    wob = D["wob"]["mars"]["wob_setpoint_n"]
    below = ph.operating_point(0.6, 0.95 * ph.z_crit, wob, 150.0, True)
    above = ph.operating_point(0.6, ph.z_crit + 0.03, wob, 150.0, True)
    assert below["torque_plug_nm"] == 0.0
    assert above["torque_total_nm"] > D["torque_fault_nm"]


H = D["head"]
LED = D["ledger"]


def test_drill_head_gear_train_and_motor() -> None:
    """16:1 = planetary x spur; planetary assembles; every mesh passes AGMA bending and contact factors."""
    cr, rot = P["criteria"], H["rotary"]
    zs, zp, zr = rot["planetary_teeth"]
    i_total = (1 + zr / zs) * rot["spur_teeth"][1] / rot["spur_teeth"][0]
    assert i_total == pytest.approx(P["actuators"]["rotary"]["gear_ratio"])
    assert (zs + zr) % rot["planets"] == 0 and zp == (zr - zs) // 2
    assert (zs + zp) * math.sin(math.pi / rot["planets"]) > zp + 2
    for mesh in rot["planetary"] + [rot["spur"], H["percussive"]["mesh"]]:
        assert mesh["fos_bending"] >= cr["gear_bending_fos"] and mesh["fos_contact"] >= cr["gear_contact_fos"]
    # Motor delivers the spindle torque limit through the train
    t_m = rot["motor_torque_nm"] * P["actuators"]["rotary"]["gear_ratio"] * P["actuators"]["rotary"]["gear_efficiency"]
    assert t_m == pytest.approx(P["actuators"]["rotary"]["max_safe_torque_nm"], rel=1e-6)
    v_r = math.pi / 4 * rot["rotor_d_m"] ** 2 * rot["rotor_l_m"]
    assert rot["motor_torque_nm"] / v_r == pytest.approx(P["drill_head"]["motor_trv_n_m_m3"], rel=1e-6)


def test_hammer_spring_and_cam_profile() -> None:
    cr, perc = P["criteria"], P["actuators"]["percussive"]
    s = D["hammer"]["spring"]
    w = P["materials"]["hammer_spring_wire"]
    k = perc["spring_rate_n_m"]
    # Rate from geometry reproduces the specified k
    k_geom = w["shear_modulus_pa"] * (s["wire_mm"] / 1e3) ** 4 / (8 * (s["coil_d_mm"] / 1e3) ** 3 * s["active_coils"])
    assert k_geom == pytest.approx(k, rel=0.01)
    assert s["active_coils"] >= cr["spring_min_active_coils"] and s["fos_fatigue"] >= cr["structural_fos_fatigue"]
    assert s["slenderness"] < 2.63 / 0.5 and s["surge_hz"] >= cr["spring_surge_ratio"] * D["hammer"]["blow_frequency_hz"]
    # Thinnest passing wire: every thinner candidate fails a check
    assert all(not r["ok"] for r in D["hammer"]["spring_sweep"] if r["wire_mm"] < s["wire_mm"])
    cam = H["cam"]
    prof = dict(cam["profile_deg_mm"])
    assert prof[0.0] == pytest.approx(0.0) and prof[cam["rise_deg"]] == pytest.approx(cam["rise_m"] * 1e3, abs=1e-3)
    assert cam["rise_deg"] + cam["dwell_deg"] == pytest.approx(360.0 / perc["cam_lobes"])
    fall_deg = 360.0 * perc["cam_speed_rps"] * LED["hammer_fall_time_ms"]["value"] / 1e3
    assert cam["dwell_deg"] >= cr["hammer_fall_time_margin"] * fall_deg
    assert cam["pressure_angle_deg"] <= cr["cam_max_pressure_angle_deg"]
    assert cam["roller_contact_pa"] <= P["materials"]["gear_steel"]["contact_strength_pa"] / cr["gear_contact_fos"]
    # Integrated cam torque agrees with the energy balance n E / (2 pi eta)
    t_mean = perc["cam_lobes"] * D["hammer"]["impact_energy_j"] / (2 * math.pi * perc["cam_efficiency"])
    assert LED["cam_torque_mean_check_nm"]["value"] == pytest.approx(t_mean, rel=1e-3)


def test_head_bearings_spline_and_interfaces() -> None:
    b, cr = H["bearings"], P["criteria"]
    assert min(b["spindle_l10_h"], b["upper_l10_h"], b["cam_l10_h"]) >= b["required_h"]
    assert min(b["spindle_s0"], b["cam_s0"]) >= cr["bearing_static_fos"]
    # WOB + spring reaction reach the spindle pair; the impact force does not
    assert b["spindle_fa_n"] < D["hammer"]["peak_force_n"] / 4
    assert H["spline"]["flank_pressure_pa"] <= P["drill_head"]["spline_allowable_pressure_pa"]
    assert LED["leadscrew_self_locking"]["value"] is True
    assert H["load_cell_n"] >= cr["load_cell_overload_factor"] * D["wob"]["mars"]["wob_limit_n"]
    assert 2 * H["load_cell_error_n"] <= P["actuators"]["z_axis_feed"]["contact_detect_wob_n"]
    st = H["stack_mm"]
    assert st["yoke_top"] <= 340.0                       # fits the carriage envelope used by the mast
    assert st["hex_top"] == 40.0                          # auger hex shank (80 mm from 40 mm below the underside)
    assert st["anvil_shoulder"] - st["nose_top"] == pytest.approx(P["drill_head"]["anvil_float_m"] * 1e3)
    assert H["plan_mm"]["leadscrew_y"] - 8.0 > H["plan_mm"]["housing_r_lower"]


def test_physical_helpers() -> None:
    # Murphy & Koop 2005: ~ 1.0e-3 Pa at 180 K... check 200 K value ~ 0.16 Pa (+/- 10 %)
    assert ice_vapour_pressure_pa(200.0) == pytest.approx(0.162, rel=0.1)
    assert mohr_coulomb_ucs_pa(1000.0, 0.0) == pytest.approx(2000.0)


def test_solar_selection_maximises_power() -> None:
    s = json.loads((DATA / "solar_trade.json").read_text())
    sel = s["selected"]
    tot = {k: v["moon"]["mean_w"] + v["mars"]["mean_w"] for k, v in s["configs"].items()}
    assert tot[sel] == max(tot.values())
    assert s["configs"]["E_track_2axis"]["moon"]["mean_w"] > s["configs"]["B_fixed_vertical_2wings"]["moon"]["mean_w"]


def test_mission_telemetry_and_energy_conservation() -> None:
    abcd = json.loads((DATA / "abcd_summary.json").read_text())
    for env_id in ("moon", "mars"):
        df = pd.read_csv(DATA / f"telemetry_{env_id}.csv")
        assert int(df.isna().sum().sum()) == 0
        assert set(df["state"].unique()) == {"IDLE", "ACTIVE", "SEARCH", "DRILL", "RETRACTION", "TRANSFER"}
        assert df["hole_bottom_m"].max() == pytest.approx(1.0, abs=1e-3)
        assert (df["wob_n"] <= df["wob_max_n"] + 1e-9).all()
        assert (df["torque_nm"] <= P["actuators"]["rotary"]["max_safe_torque_nm"] + 1e-9).all()
        assert (df["bit_temp_k"] <= df["crit_temp_k"] + 1e-6).all()
        a = abcd["environments"][env_id]
        e = df["power_w"].sum() / 3600.0
        assert a["D_total_mission_energy_wh"] == pytest.approx(e, abs=0.06)
        assert a["D_peak_electrical_power_w"] >= a["D_mean_electrical_power_w"]
    for env_id in ("moon", "mars"):
        off = abcd["off_nominal"][env_id]
        assert off["A_reached_depth_m"] == pytest.approx(1.0, abs=1e-3)
    assert sum(abcd["off_nominal"][e]["num_faults"] for e in ("moon", "mars")) >= 1


def test_cad_exports_and_scene_graph_nodes() -> None:
    cad = PROJECT_ROOT / "exports" / "cad"
    for name in ("auger_bit", "drill_bit_crown", "drill_head", "drill_head_section", "mast_and_brush", "mast_stowed",
                 "brush_station_exploded", "power_system", "full_drill_rig"):
        for ext in ("step", "stl", "glb"):
            f = cad / f"{name}.{ext}"
            assert f.exists() and f.stat().st_size > 1000
        assert (PROJECT_ROOT / "docs_src" / "assets" / "cad" / f"{name}.glb").exists()
        m = trimesh.load_mesh(str(cad / f"{name}.stl"), process=False)
        assert abs(float(m.mass_properties["volume"])) > 1.0  # mm^3
    nodes = set(trimesh.load(str(cad / "full_drill_rig.glb")).graph.nodes)
    assert {"Frame_Node", "Solar_Node", "Carriage_Node", "Auger_Node", "Hammer_Cam_Node", "Cam_Ring_Node", "Brush_Gear_Node",
            "Solar_Wing_Port_Az_Node", "Solar_Wing_Port_Elev_Node"}.issubset(nodes)
    man = json.loads((DATA / "cad_manifest.json").read_text())
    assert man["layout"]["sampling_flute_depth_mm"] == D["auger"]["flute_depth_mm"]
    assert man["layout"]["sampling_section_length_mm"] == D["auger"]["sampling_length_mm"]
    # Head parts sit in the right kinematic groups
    assert {"Head_Housing_Ti", "WOB_Load_Cell_500N", "Spindle_Bearing_7004ACD_A_Rings"}.issubset(man["nodes"]["Carriage_Node"])
    assert {"Anvil", "Splined_Hollow_Spindle", "Hammer_Compression_Spring"}.issubset(man["nodes"]["Auger_Node"])
    assert "Hammer_Mass" in man["nodes"]["Hammer_Cam_Node"] and "Barrel_Cam_Track_2_Lobe" in man["nodes"]["Cam_Ring_Node"]


def test_compiled_quarto_website_outputs() -> None:
    docs = PROJECT_ROOT / "docs"
    for page in ("index.html", "digital_twin.html", "theory_and_sizing.html", "cad_architecture.html",
                 "autonomous_control.html"):
        assert (docs / page).exists()
    for asset in ("assets/cad/full_drill_rig.glb", "assets/data/kinematic_keyframes.json", "assets/js/drill_kinematics.js"):
        assert (docs / asset).exists()


# ------------------------------------------------------------------------------
# Phase 3: auger string and brushing station
# ------------------------------------------------------------------------------
S = D["string"]
B = D["brush"]


def test_string_fe_matches_closed_form() -> None:
    """Uniform clamped-pinned column: P_cr = pi^2 EI/(0.6992 L)^2, f1 = 3.9266^2/(2 pi) sqrt(EI/(m L^4))."""
    from src.design_string import StringFE
    ei, m, L = 171.0, 0.37, 1.2
    fe = StringFE([(0.0, L + 1e-9, {"EI": ei, "m": m, "GJ": 134.0, "rhoJ": 1e-5})], L, 120)
    r = fe.lateral([0.0])
    assert r["p_cr_n"] == pytest.approx(math.pi ** 2 * ei / (0.69915 * L) ** 2, rel=2e-3)
    assert r["f1_hz"] == pytest.approx(3.9266 ** 2 / (2 * math.pi) * math.sqrt(ei / (m * L ** 4)), rel=2e-3)
    # Clamped-free shaft: f = c / (4 L), c = sqrt(GJ / rhoJ)
    assert fe.torsion()["f1_hz"] == pytest.approx(math.sqrt(134.0 / 1e-5) / (4 * L), rel=2e-3)


def test_string_buckling_frequencies_and_windup() -> None:
    bm, cr = S["beam"], P["auger_string"]
    f_blow = D["hammer"]["blow_frequency_hz"]
    assert bm["p_cr_n"] >= P["criteria"]["structural_fos_yield"] * bm["p_design_n"]
    assert bm["p_design_n"] == max(w["wob_limit_n"] for w in D["wob"].values())
    # With the guide bushing every lateral mode over the stroke clears the blow-rate band; torsion far above
    assert all(r["f1_hz"] >= (1 + cr["resonance_separation"]) * f_blow for r in bm["sweep"])
    assert bm["torsion_f1_hz"] >= (1 + cr["resonance_separation"]) * f_blow
    assert bm["f1_hz"] > P["actuators"]["rotary"]["max_rpm"] / 60
    assert bm["f1_hz"] > bm["f1_no_bushing_hz"] and bm["p_cr_n"] > bm["p_cr_no_bushing_n"]
    # Wind-up is mostly in the slender sampling root: twist > that of the 14/12 tube alone over the full length
    g = P["materials"]["auger_steel"]["youngs_modulus_pa"] / (2 * (1 + P["materials"]["auger_steel"]["poisson"]))
    a = P["auger_geometry"]
    j_tube = math.pi / 32 * (a["tube_outer_diameter_m"] ** 4 - a["tube_inner_diameter_m"] ** 4)
    t_max = P["actuators"]["rotary"]["max_safe_torque_nm"]
    assert bm["windup_rad"] > t_max * P["cad_layout"]["auger_length_m"] / (g * j_tube)
    assert LED["string_windup_deg"]["value"] == pytest.approx(math.degrees(bm["windup_rad"]), abs=0.01)


def test_flight_attachment_and_drive_interface() -> None:
    fl, sh, cr = S["flight"], S["shank"], P["criteria"]
    if fl["attachment"] == "integral":
        assert fl["integral_fos"] >= cr["structural_fos_fatigue"]
        assert fl["weld_life"] < fl["required_cycles"]          # welds rejected for a reason
    else:
        assert fl["inertia_mpa"] <= fl["cafl_mpa"] and fl["weld_mpa"] <= fl["weld_allow_mpa"]
    t_wall = (P["auger_geometry"]["tube_outer_diameter_m"] - P["auger_geometry"]["tube_inner_diameter_m"]) / 2
    assert sh["lap_m"] >= P["auger_string"]["braze_lap_to_wall"] * t_wall - 1e-12
    assert sh["flank_mpa"] * 1e6 <= P["drill_head"]["spline_allowable_pressure_pa"]
    sy = P["materials"]["auger_steel"]["yield_strength_pa"] / 1e6
    assert max(sh["impact_mpa"], sh["torsion_mpa"] * math.sqrt(3), sh["pin_mpa"] * math.sqrt(3)) * cr["structural_fos_yield"] <= sy
    # Interface with the drill head: hex sits from the anvil nose to hex_top, socket 0.1 mm over the shank
    assert sh["engagement_m"] * 1e3 == H["stack_mm"]["hex_top"] - H["stack_mm"]["anvil_bottom"]
    assert sh["hex_r_m"] * 1e3 < 8.1
    # Joint stack in the beam model matches the CAD bit
    from src.cad_bit import bit_top_z
    assert S["stations_m"]["tube_bottom"] * 1e3 == pytest.approx(bit_top_z(P, D, 0.0))


def test_brush_worm_wheel_mesh() -> None:
    m = B["mesh"]
    rows = m["rows"]
    assert rows[-1]["n"] == m["tufts"] and rows[-1]["interference"] <= 0 and rows[-1]["hub_fit"] >= 0
    assert all(r["interference"] > 0 or r["hub_fit"] < 0 for r in rows[:-1])   # smallest tuft count that works
    p_s = D["auger"]["sampling_pitch_mm"] / 1e3
    assert 2 * math.pi * m["r_p_m"] == pytest.approx(m["tufts"] * p_s)
    r_od = P["auger_geometry"]["outer_diameter_m"] / 2
    r_root = D["auger"]["root_od_mm"] / 2e3
    assert m["axle_offset_m"] == pytest.approx(m["r_p_m"] + 0.5 * (r_od + r_root))
    assert m["axle_offset_m"] - m["r_hub_m"] >= r_od                            # hub clears the flights
    assert m["axle_offset_m"] - m["r_tip_m"] < r_root                            # tips reach past the root


def test_brush_bristles_force_budget_and_drag() -> None:
    br, cr = B["bristle"], P["criteria"]
    assert br["fos"] >= cr["structural_fos_fatigue"]
    assert br["contact_n"] <= br["budget_n"]
    assert br["tip_pressure_pa"] >= br["passive_pa"]
    ok = [r for r in br["sweep"] if r["ok"]]
    assert ok and max(r["wire_mm"] for r in ok) == pytest.approx(br["wire_m"] * 1e3)   # thickest passing wire
    dr = B["drag"]
    assert dr["total_nm"] == pytest.approx(dr["plough_nm"] + dr["friction_nm"] + dr["bushing_nm"])
    assert "brush_drag_torque_nm" not in P["cutting_model"]                     # no longer a coefficient
    # The state machine uses the derived drag
    df = pd.read_csv(DATA / "telemetry_moon.csv")
    assert df.loc[df["state"] == "TRANSFER", "torque_nm"].round(4).eq(round(dr["total_nm"], 4)).all()
    # Spring preload is the required bristle force; spring stress within the static allowable
    sp = B["arm"]["spring"]
    assert sp["moment_nm"] == pytest.approx(br["contact_n"] * B["arm"]["length_m"])
    assert sp["stress_mpa"] * 1e6 * cr["structural_fos_yield"] <= P["materials"]["hammer_spring_wire"]["allowable_shear_pa"] / 0.577
    assert 4.0 <= sp["index"] <= 12.0


def test_chute_and_cup_sizing() -> None:
    bs, ch, cup = P["actuators"]["brush_station"], B["chute"], B["cup"]
    assert ch["angle_deg"] >= math.degrees(math.atan(bs["cuttings_bronze_friction_coeff"])) + bs["chute_angle_margin_deg"]
    v_bite = P["system"]["bite_depth_m"] * D["auger"]["hole_area_mm2"] / 1e6 * P["cutting_model"]["cuttings_bulking_factor"]
    assert math.pi / 4 * cup["id_m"] ** 2 * cup["depth_m"] >= v_bite * P["criteria"]["cuttings_retention_margin"] * (1 - 1e-6)
    assert ch["flow_depth_m"] < ch["wall_h_m"]
    assert ch["z_out_m"] > cup["z0_m"] + cup["depth_m"]                         # outlet above the cup rim
    man = json.loads((DATA / "cad_manifest.json").read_text())
    k = man["kinematics"]
    assert k["brush_radius_m"] == pytest.approx(B["mesh"]["r_p_m"])
    assert k["brush_axle_m"][2] == pytest.approx(B["mesh"]["axle_offset_m"])        # glTF +Z = CAD -Y (fore)
    assert (PROJECT_ROOT / "exports" / "cad" / "brush_station_exploded.glb").exists()


def test_sample_carousel_one_sealed_cup_per_bite() -> None:
    ca, cup, car = P["actuators"]["sample_carousel"], B["cup"], B["carousel"]
    bites = math.ceil(round(P["system"]["target_depth_m"] / P["system"]["bite_depth_m"], 6))
    assert car["cups"] == bites and car["stations"] == bites + ca["blank_stations"]
    # Cups fit round the pitch circle and clear the guide-bushing flange; turntable clears the ground
    pitch = 2 * math.pi * car["pitch_radius_m"] / car["stations"]
    assert pitch >= car["flange_d_m"] + ca["cup_gap_m"] - 1e-9
    assert cup["y_m"] - car["flange_d_m"] / 2 >= D["string"]["bushing"]["bore_m"] / 2 + 0.009 + ca["radial_clearance_m"] - 1e-9
    assert car["turntable_z0_m"] >= ca["ground_clearance_m"] - 1e-9
    assert cup["rim_m"] < P["cad_layout"]["foot_top_m"] - P["drill_mast"]["foot_plate_thickness_m"]   # sealed under the plate
    # Drive: ECSS-motorised torque within the motor rating; index done within every bite's descent
    assert car["index_torque_nm"] <= ca["index_motor"]["rated_torque_nm"]
    for env_id in ("moon", "mars"):
        df = pd.read_csv(DATA / f"telemetry_{env_id}.csv")
        st = df["carousel_station"].to_numpy()
        assert st[0] == 0.0 and st[-1] == pytest.approx(car["stations"])                 # blank under the port at both ends
        assert np.all(np.diff(st) >= -1e-9)                                               # only ever indexes forward
        tr = df[df["state"] == "TRANSFER"]
        assert np.allclose(tr["carousel_station"], tr["bite_number"])                     # bite n always fills cup n
    man = json.loads((DATA / "cad_manifest.json").read_text())
    nodes = man["nodes"]["Carousel_Node"]
    assert sum(n.startswith("Sample_Cup_") for n in nodes) == bites and "Carousel_Turntable" in nodes
    assert man["kinematics"]["carousel"]["stations"] == car["stations"]


# ------------------------------------------------------------------------------
# Phase 4: deployable mast and lander interface
# ------------------------------------------------------------------------------
MA = D["mast"]


def test_mast_fe_matches_closed_form() -> None:
    """Pinned-pinned f1 = (pi/2) sqrt(EI/(m L^4)); cantilever tip load w = P L^3 / (3 EI); propped-cantilever reaction."""
    from src.design_mast import Beam
    ei, m, L = 2.0e4, 3.0, 1.5
    bm = Beam(0.0, L, 60, lambda s: (ei, m, 0.05))
    assert bm.f1([(0.0, "pin"), (L, "pin")]) == pytest.approx(math.pi / 2 * math.sqrt(ei / (m * L ** 4)), rel=1e-3)
    st = bm.static([(0.0, "clamp")], loads=[(L, 10.0, 0.0)])
    assert st["w"][-1] == pytest.approx(10.0 * L ** 3 / (3 * ei), rel=1e-6)
    assert st["sig_over_e"] * ei == pytest.approx(10.0 * L * 0.05, rel=1e-6)          # sigma/E x EI = M c
    # Clamped-pinned (propped) uniform load: prop reaction 3 w L / 8
    st = bm.static([(0.0, "clamp"), (L, "pin")], accel=1.0)
    assert abs(st["react"][2 * bm.node(L)]) == pytest.approx(3 * m * L / 8, rel=1e-3)
    # Miles 3-sigma on the GEVS plateau
    from src.design_mast import miles_3sigma_g, psd_at
    assert psd_at(P, 100.0) == pytest.approx(0.16)
    assert psd_at(P, 35.0) == pytest.approx(0.026 * (35.0 / 20.0) ** (math.log(0.16 / 0.026) / math.log(2.5)))
    assert miles_3sigma_g(P, 100.0) == pytest.approx(3 * math.sqrt(math.pi / 2 * 100 * P["criteria"]["random_vibration_q"] * 0.16))


def test_mast_modes_alignment_and_buckling() -> None:
    sep = MA["separation"]
    for f in MA["f_deployed_hz"]:                       # every drilling pose outside every forcing band
        assert not any(lo / (1 + sep) <= f <= hi * (1 + sep) for lo, hi in MA["forcing_hz"])
    assert MA["f_stowed_hz"] >= P["criteria"]["launch_min_frequency_hz"]
    assert MA["alignment_m"] <= P["auger_string"]["guide_bushing_radial_clearance_m"]
    assert MA["buckling_n"] >= P["criteria"]["structural_fos_ultimate"] * MA["compression_n"]
    assert MA["launch_stress_mpa"] * 1e6 * P["criteria"]["structural_fos_yield"] <= MA["yield_pa"]
    assert MA["launch_stress_mpa"] * 1e6 * P["criteria"]["structural_fos_ultimate"] <= MA["ultimate_pa"]
    # Rail spans clear the launch minimum on their own; an end-supported rail would not
    assert MA["rail_span_m"] <= MA["rail_span_max_m"] + 1e-9
    assert LED["rail_span_max_mm"]["value"] / 1e3 < MA["geometry_m"]["z_crosshead_bottom"] - MA["geometry_m"]["z_foot_top"]
    # The chosen design is the lightest feasible one and the design load is the lift-off WOB
    assert MA["wob_design_n"] == max(w["wob_limit_n"] for w in D["wob"].values())
    assert 0 < MA["candidates_feasible"] < MA["candidates_total"]


def test_leadscrew_and_stowed_secondary_members() -> None:
    ls, stw, cr = MA["leadscrew"], MA["stow"], P["criteria"]
    assert ls["p_cr_n"] >= cr["structural_fos_ultimate"] * MA["wob_design_n"]
    assert ls["n_max_rpm"] <= cr["leadscrew_speed_fraction"] * ls["n_cr_rpm"]
    assert ls["nut_launch_pa"] <= P["drill_head"]["nut_allowable_pressure_pa"]
    # Tr16x4 minor diameter (ISO 2904): d3 = d - 2 (P/2 + a_c)
    assert ls["d3_m"] == pytest.approx(0.0115)
    # The stow pose makes the screw and string far stiffer than leaving the carriage at home
    assert stw["screw_f1_hz"] > 2 * stw["screw_home_f1_hz"] and stw["string_f1_hz"] > 2 * stw["string_home_f1_hz"]
    assert stw["screw_stress_mpa"] * 1e6 * cr["structural_fos_yield"] <= P["materials"]["feed_steel"]["yield_strength_pa"]
    assert stw["screw_sag_mm"] < stw["screw_gap_mm"]
    assert stw["string_stress_mpa"] * 1e6 * cr["structural_fos_yield"] <= P["materials"]["auger_steel"]["yield_strength_pa"]
    assert stw["tip_z_m"] < D["string"]["bushing"]["z0_m"]            # the bit is through the guide bushing at launch


def test_hinge_latch_hold_down_and_actuator() -> None:
    cr, mc = P["criteria"], P["drill_mast"]
    hp, lp = MA["hinge_pin"], MA["latch_pin"]
    assert hp["fos_shear"] >= cr["structural_fos_ultimate"] and hp["fos_bearing"] >= cr["structural_fos_yield"]
    assert lp["fos_shear"] >= cr["structural_fos_ultimate"]
    a_s = mc["hold_down_bolts"][MA["hold_down_bolt"]]
    pre = mc["hold_down_preload_fraction"] * P["materials"]["bolt_a286"]["yield_strength_pa"] * a_s
    assert pre >= cr["joint_separation_fos"] * cr["structural_fos_ultimate"] * MA["hold_down_tension_n"]
    dep = MA["deploy"]
    assert dep["t_req_nm"] >= cr["motorisation_static_factor"] * dep["t_grav_nm"]
    g_mars = P["environments"]["mars"]["gravity_m_s2"]
    assert dep["t_grav_nm"] == pytest.approx(dep["mass_kg"] * g_mars * dep["arm_m"])
    assert dep["arm_m"] >= abs(dep["cg_dz_m"]) - 1e-9                      # the worst case is the mast lying horizontal


def test_lander_tilt_keeps_wob_rule_and_reacts_torque() -> None:
    """WOB set-point stays m g / FOS; a side-mounted drill tilts the lander before it lifts, so the legs are sized for that."""
    la, m, fos = MA["lander"], P["system"]["lander_mass_kg"], P["criteria"]["wob_lift_off_fos"]
    for env_id, env in P["environments"].items():
        e = la["env"][env_id]
        assert D["wob"][env_id]["wob_setpoint_n"] == pytest.approx(m * env["gravity_m_s2"] / fos, abs=0.01)
        assert e["wob_tip_n"] < m * env["gravity_m_s2"]                          # tilt comes first
        assert e["wob_tip_n"] >= D["wob"][env_id]["wob_setpoint_n"] + H["load_cell_error_n"] - 1e-6
        assert e["near_leg_n"] > 0 and e["far_leg_n"] > 0
        assert e["near_leg_n"] * 2 + e["far_leg_n"] * 2 == pytest.approx(m * env["gravity_m_s2"] - D["wob"][env_id]["wob_setpoint_n"])
        assert e["torque_fos"] >= fos
    assert la["tilt_ratio"] == pytest.approx((la["far_leg_y_m"] - la["cg_y_m"]) / la["far_leg_y_m"])
    assert la["m_rest_kg"] > 0


def test_solar_and_cad_use_mast_geometry() -> None:
    from src.solar import array_geometry
    geo = array_geometry(P)
    assert geo["wing_x_m"] == pytest.approx(MA["solar"]["wing_x_m"])
    assert geo["boxes_m"][0][1][2] == pytest.approx(MA["geometry_m"]["z_motor_top"])       # mast envelope reaches the motor top
    man = json.loads((DATA / "cad_manifest.json").read_text())
    assert man["layout"]["mast_top_mm"] == pytest.approx(MA["geometry_m"]["z_top"] * 1e3)
    assert man["layout"]["wing_separation_mm"] == pytest.approx(2e3 * MA["geometry_m"]["wing_x"])
    assert f"Mast_Spine_{MA['section_mm'][0]:.0f}x{MA['section_mm'][1]:.0f}x{MA['section_mm'][2]:.0f}" in man["nodes"]["Frame_Node"]
    assert sum(n.startswith("Rail_Bracket_") for n in man["nodes"]["Frame_Node"]) == MA["n_brackets"]
    # Stowed GLB: the mast lies over the deck (low, long in y), deployed rig stands to the mast top
    st = man["assemblies"]["mast_stowed"]["bbox_mm"]
    dep = man["assemblies"]["mast_and_brush"]["bbox_mm"]
    assert st[1][2] < 0.5 * dep[1][2]
    assert st[1][1] - st[0][1] > MA["geometry_m"]["z_top"] * 1e3 - MA["geometry_m"]["z_hinge"] * 1e3


# ------------------------------------------------------------------------------
# Phase 5: portfolio showcase (renders, drawings, handout, landing page)
# ------------------------------------------------------------------------------
def test_portfolio_renders_drawings_and_handout() -> None:
    from PIL import Image
    from src.render import VIEWS
    from src.showcase import benchmark_rows, load

    docs = PROJECT_ROOT / "docs"
    for name, (_, _, _, w, h, _, _) in VIEWS.items():
        img = Image.open(docs / "assets" / "renders" / f"{name}.png").convert("L")
        assert img.size == (w, h)
        # not blank: shaded model pixels (darker than the background gradient) cover part of the frame
        hist = img.histogram()
        assert sum(hist[:200]) > 0.02 * w * h, name
    for stem in ("DRL-001_drill_bit", "DRL-002_drill_head"):
        svg = (docs / "assets" / "drawings" / f"{stem}.svg").read_text()
        assert stem.split("_")[0] in svg and "<path" in svg
        assert (docs / "assets" / "drawings" / f"{stem}.pdf").read_bytes()[:4] == b"%PDF"
    pdf = (docs / "assets" / "handout" / "design_summary.pdf").read_bytes()
    assert pdf[:4] == b"%PDF" and pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages") == 1  # one page

    # Landing page shows the live benchmark numbers and links every deliverable
    html = (docs / "index.html").read_text()
    for r in benchmark_rows(load()):
        assert r["fmt"].format(r["moon"]) in html and r["fmt"].format(r["mars"]) in html
    for link in ("assets/handout/design_summary.pdf", "assets/drawings/DRL-001_drill_bit.pdf",
                 "assets/drawings/DRL-002_drill_head.pdf", "assets/renders/rig_deployed.png", "digital_twin.html"):
        assert link in html
    # no build-machine paths or source-file names leak into the rendered prose
    for page in docs.glob("*.html"):
        text = page.read_text()
        assert "/Users/" not in text and "system_params.yaml" not in text, page.name
