"""
Module 0: First-principles design derivations.

Turns the primary inputs in config/system_params.yaml into every derived design
quantity used downstream (physics, state machine, CAD, website). Each derivation
is recorded with its governing equation, inputs and result in
exports/data/design_derivations.json so it can be audited.

Derived here:
  * WOB limits           : lift-off limit m*g and set-point m*g / FOS
  * Hammer dynamics      : blow energy, hammer velocity, impact force/stress, blow rate,
                           minimum WOB to keep the bit seated, spring stress, natural frequency
  * Auger root & flutes  : smallest root diameter passing Goodman fatigue (impact + torsion)
                           -> sampling flute depth
  * Sampling pitch       : static retention of cuttings on the flight (helix angle < friction angle)
  * Sampling length      : flute volume >= bulked cuttings of one bite x margin
  * Transport pitch      : screw-conveying efficiency optimum alpha = 45 deg - phi/2
  * Conveying check      : centrifugal lift condition mu*omega^2*r > g
  * Critical temperatures: Moon - ice sublimation loss criterion (Murphy & Koop 2005 vapour pressure)
                           Mars - melt margin below 273.15 K
  * Actuator warm-up time: lumped heat balance with radiative / convective losses
  * Drill head (src/design_head.py): hammer spring, motor + 16:1 gear train, barrel cam, spline,
                           catalogue bearing lives, load cell, leadscrew nut, axial stack
  * Auger string (src/design_string.py): flight root + weld, braze, hex shank, buckling, natural
                           frequencies, wind-up, thermocouple path
  * Brushing station (src/design_brush.py): worm-wheel mesh, bristles, drag torque, spring arm,
                           sweep speed, chute, cup, stow position
  * Deployable mast (src/design_mast.py): spine section, latch, stow pose, hold-downs, hinge, actuator,
                           leadscrew, lander deck, legs, tilt limit, solar-wing stations
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.design_brush import derive_brush
from src.design_head import derive_head, derive_spring
from src.design_mast import derive_mast
from src.design_string import derive_string

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "system_params.yaml"
EXPORTS_DATA_DIR = PROJECT_ROOT / "exports" / "data"

SIGMA_SB = 5.670374419e-8
R_GAS = 8.314462618
M_H2O = 0.01801528


def load_system_params(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Ledger:
    """Collects derivations as {key: {value, unit, equation, note}}."""

    def __init__(self) -> None:
        self.items: Dict[str, Dict[str, Any]] = {}

    def add(self, key: str, value: Any, unit: str, equation: str, note: str = "") -> Any:
        self.items[key] = {"value": value, "unit": unit, "equation": equation, "note": note}
        return value


# ------------------------------------------------------------------------------
# Thermophysics helpers
# ------------------------------------------------------------------------------
def ice_vapour_pressure_pa(t_k: float) -> float:
    """Murphy & Koop (2005) Eq. 7, water-ice saturation vapour pressure (valid T > 110 K)."""
    return math.exp(9.550426 - 5723.265 / t_k + 3.53068 * math.log(t_k) - 0.00728332 * t_k)


def ice_sublimation_flux(t_k: float) -> float:
    """Hertz-Knudsen free sublimation flux into vacuum (kg m^-2 s^-1)."""
    return ice_vapour_pressure_pa(t_k) * math.sqrt(M_H2O / (2.0 * math.pi * R_GAS * t_k))


def mohr_coulomb_ucs_pa(cohesion_pa: float, phi_deg: float) -> float:
    """UCS of a Mohr-Coulomb material: 2 c tan(45 + phi/2)."""
    return 2.0 * cohesion_pa * math.tan(math.radians(45.0 + 0.5 * phi_deg))


def layer_ucs_pa(layer: Dict[str, Any]) -> float:
    if layer.get("ucs_mpa") is not None:
        return float(layer["ucs_mpa"]) * 1e6
    return mohr_coulomb_ucs_pa(float(layer["cohesion_kpa"]) * 1e3, float(layer["friction_angle_deg"]))


# ------------------------------------------------------------------------------
# Derivations
# ------------------------------------------------------------------------------
def derive_wob(p: Dict[str, Any], led: Ledger) -> Dict[str, Dict[str, float]]:
    m = float(p["system"]["lander_mass_kg"])
    fos = float(p["criteria"]["wob_lift_off_fos"])
    out = {}
    for env_id, env in p["environments"].items():
        g = float(env["gravity_m_s2"])
        lim = led.add(f"wob_lift_off_{env_id}_n", round(m * g, 2), "N", "WOB_lim = m g",
                      f"lander lift-off limit, {env_id}")
        sp = led.add(f"wob_setpoint_{env_id}_n", round(m * g / fos, 2), "N", "WOB_set = m g / FOS",
                     f"FOS = {fos}")
        out[env_id] = {"wob_limit_n": lim, "wob_setpoint_n": sp}
    return out


def derive_hammer(p: Dict[str, Any], led: Ledger, hammer_mass_kg: float, a_tube: float) -> Dict[str, float]:
    perc = p["actuators"]["percussive"]
    steel = p["materials"]["auger_steel"]
    k = float(perc["spring_rate_n_m"])
    h = float(perc["cam_lift_m"])
    e_imp = led.add("impact_energy_j", round(0.5 * k * h * h, 4), "J", "E = 1/2 k h^2",
                    "spring energy released per blow")
    f_blow = led.add("blow_frequency_hz", float(perc["cam_lobes"]) * float(perc["cam_speed_rps"]), "Hz",
                     "f = n_lobes x cam_rps")
    v_h = led.add("hammer_impact_velocity_m_s", round(math.sqrt(2 * e_imp / hammer_mass_kg), 3), "m/s",
                  "v = sqrt(2E/m_h)", f"m_h = {hammer_mass_kg:.3f} kg from CAD hammer geometry")
    c = math.sqrt(float(steel["youngs_modulus_pa"]) / float(steel["density_kg_m3"]))
    hd = p["drill_head"]
    a_h = math.pi / 4 * (float(hd["hammer_od_m"]) ** 2 - float(hd["spline_major_m"]) ** 2)
    z_h = float(steel["density_kg_m3"]) * c * a_h
    z_t = float(steel["density_kg_m3"]) * c * a_tube
    f_peak = led.add("impact_peak_force_n", round(v_h * z_h * z_t / (z_h + z_t), 1), "N",
                     "F = v Z_h Z_r / (Z_h + Z_r),  Z = rho c A", "1-D stress-wave impact (Lundberg 1982)")
    fn = led.add("hammer_spring_natural_hz", round(math.sqrt(k / hammer_mass_kg) / (2 * math.pi), 1), "Hz",
                 "f_n = sqrt(k/m_h) / 2pi", "must exceed 2 x blow rate so the hammer re-seats each cycle")
    e_rest = float(perc["coefficient_of_restitution"])
    w_min = led.add("percussive_min_wob_n", round(hammer_mass_kg * v_h * (1 + e_rest) * f_blow, 1), "N",
                    "W_min = m_h v (1+e) f", "mean impulse flux the bit must react to stay seated")
    spring = derive_spring(p, led, k, h, f_blow)
    tau_s = spring["tau_static_mpa"]
    p_mech = led.add("percussive_mech_power_w", round(e_imp * f_blow, 2), "W", "P = E f")
    eta = float(perc["motor_efficiency"]) * float(perc["cam_efficiency"])
    p_el = led.add("percussive_elec_power_w", round(e_imp * f_blow / eta, 2), "W", "P_el = E f / (eta_m eta_cam)")
    return {"impact_energy_j": e_imp, "blow_frequency_hz": f_blow, "impact_velocity_m_s": v_h,
            "peak_force_n": f_peak, "natural_hz": fn, "min_wob_n": w_min, "spring_shear_mpa": tau_s,
            "mech_power_w": p_mech, "elec_power_w": p_el, "efficiency": eta, "mass_kg": hammer_mass_kg,
            "spring": {k2: v2 for k2, v2 in spring.items() if k2 != "rows"}, "spring_sweep": spring["rows"]}


def derive_bit(p: Dict[str, Any], led: Ledger, hammer: Dict[str, float]) -> Dict[str, Any]:
    """
    Bit crown design from contact mechanics, cutting geometry, continuity and joint strength.

    Edge radius  : Hertz line contact (cylinder on flat). Design case = a blow landing on a hard clast
                   larger than the hole, impact shared by both edges. Smallest radius (0.25 mm steps) that
                   keeps p_max <= sigma_c,WC / FOS; a smaller radius = shorter wearflat = lower rotary threshold.
                   p_max = sqrt(w E* / (pi R)),  w = F / (n L)
    Crushing     : the same edge on the hardest drilled layer must exceed crushing_ratio x UCS.
    Wedge angle  : beta = 90 - clearance - rake.
    Indexing     : rotation between blows theta = 360 N / (60 f); arc spacing at gauge s = a theta.
    Junk slots   : total slot area >= one sampling-flute channel area (slots never the bottleneck).
    Braze        : shear from the tangential cutting force over the braze area.
    Joint        : shoulder bearing stress under the impact force; thread stripping under torque is checked
                   via the torque-to-shear relation tau = 2 T / (pi d_p^2 L_e).
    """
    b = p["drill_bit"]
    a = p["auger_geometry"]
    gauge_r = float(a["gauge_diameter_m"]) / 2
    n_c = int(a["num_cutters"])
    edge_len = gauge_r - 0.0005            # each insert runs from 0.5 mm off the axis to the gauge
    f_imp = float(hammer["peak_force_n"])
    w_line = f_imp / (n_c * edge_len)

    def e_star(e1, n1, e2, n2):
        return 1.0 / ((1 - n1 ** 2) / e1 + (1 - n2 ** 2) / e2)

    e_wc, n_wc = float(b["insert_youngs_modulus_pa"]), float(b["insert_poisson"])
    es_clast = e_star(e_wc, n_wc, float(b["clast_youngs_modulus_pa"]), float(b["clast_poisson"]))
    p_allow = float(b["insert_compressive_strength_pa"]) / float(b["carbide_contact_fos"])
    r_min = w_line * es_clast / (math.pi * p_allow ** 2)
    r_e = math.ceil(r_min / 0.00025) * 0.00025
    p_clast = math.sqrt(w_line * es_clast / (math.pi * r_e))
    led.add("bit_edge_line_load_n_mm", round(w_line / 1e3, 1), "N/mm", "w = F_imp / (n_c L_edge)")
    led.add("bit_edge_radius_mm", round(r_e * 1e3, 2), "mm", "R >= w E* / (pi (sigma_c/FOS)^2)",
            f"Hertz line contact on a basalt clast; minimum {r_min*1e3:.2f} mm rounded up to 0.25 mm")
    led.add("bit_edge_contact_fos", round(float(b["insert_compressive_strength_pa"]) / p_clast, 2), "-",
            "sigma_c,WC / p_max", f"p_max = {p_clast/1e6:.0f} MPa on the clast")

    # Crushing check on the hardest drilled layer (UCS from the stratigraphy)
    ucs_max = max(layer_ucs_pa(L) for env in p["environments"].values() for L in env["stratigraphy"])
    e_rock = float(b["rock_modulus_ratio"]) * ucs_max
    p_rock = math.sqrt(w_line * e_star(e_wc, n_wc, e_rock, float(b["rock_poisson"])) / (math.pi * r_e))
    led.add("bit_crushing_ratio", round(p_rock / ucs_max, 1), "-", "p_max(rock) / UCS_max",
            f"must exceed {b['crushing_ratio_min']}; hardest layer UCS {ucs_max/1e6:.0f} MPa")

    rake, clr = float(b["rake_angle_deg"]), float(b["clearance_angle_deg"])
    wedge = led.add("bit_wedge_angle_deg", 90.0 - clr - rake, "deg", "beta = 90 - alpha - gamma",
                    f"rake {rake} deg, clearance {clr} deg")

    # Indexing between blows at nominal RPM
    rpm = float(p["actuators"]["rotary"]["nominal_rpm"])
    theta = 360.0 * rpm / 60.0 / float(hammer["blow_frequency_hz"])
    led.add("bit_index_angle_deg", round(theta, 1), "deg", "theta = 360 N / (60 f)", "rotation between hammer blows")
    led.add("bit_index_spacing_gauge_mm", round(gauge_r * math.radians(theta) * 1e3, 2), "mm", "s = a theta",
            "each blow strikes fresh rock this far from the previous crater at the gauge")

    # Junk slots: continuity with one sampling-flute channel
    body_r = gauge_r - float(b["body_under_gauge_m"]) - 0.0005
    t_ins = float(b["insert_thickness_m"])
    slot_inner = t_ins + 0.0005
    slot_depth = body_r - slot_inner
    return {"edge_radius_m": r_e, "wedge_deg": wedge, "index_deg": theta, "body_r_m": body_r,
            "slot_inner_m": slot_inner, "slot_depth_m": slot_depth, "line_load_n_m": w_line,
            "p_clast_pa": p_clast, "p_rock_pa": p_rock}


def finish_bit(p: Dict[str, Any], led: Ledger, bit: Dict[str, Any], auger: Dict[str, Any], hammer: Dict[str, float]) -> None:
    """Checks that need the auger result (flute channel, root) and the torque limit."""
    b = p["drill_bit"]
    channel = (float(p["auger_geometry"]["outer_diameter_m"]) / 2 - auger["root_od_mm"] / 2e3) * (
        auger["sampling_pitch_mm"] / 1e3 - float(p["auger_geometry"]["sampling_flight_thickness_m"]))
    width = channel / (2 * bit["slot_depth_m"])
    width = math.ceil(width / 0.0005) * 0.0005
    bit["slot_width_m"] = width
    led.add("bit_junk_slot_width_mm", round(width * 1e3, 1), "mm", "2 w d >= A_channel",
            f"2 slots x {bit['slot_depth_m']*1e3:.1f} mm deep; flute channel {channel*1e6:.0f} mm^2")
    t_lim = float(p["actuators"]["rotary"]["max_safe_torque_nm"])
    r_mid = float(p["auger_geometry"]["gauge_diameter_m"]) / 4
    f_t = t_lim / (int(p["auger_geometry"]["num_cutters"]) * r_mid)
    braze_area = float(b["insert_height_m"]) * 0.6 * (float(p["auger_geometry"]["gauge_diameter_m"]) / 2)
    led.add("bit_braze_fos", round(float(b["braze_shear_strength_pa"]) / (f_t / braze_area), 0), "-",
            "tau_braze / (F_t / A_braze)", f"tangential force {f_t:.0f} N per insert at the torque limit")
    d_maj, pitch, le = float(b["joint_thread_major_m"]), float(b["joint_thread_pitch_m"]), float(b["joint_engagement_m"])
    d_p = d_maj - 0.6495 * pitch
    a_sh = math.pi / 4 * (float(b["joint_shoulder_od_m"]) ** 2 - d_maj ** 2)
    sy = float(p["materials"]["auger_steel"]["yield_strength_pa"])
    led.add("bit_joint_shoulder_stress_mpa", round(hammer["peak_force_n"] / a_sh / 1e6, 1), "MPa", "F_imp / A_shoulder",
            f"FOS {sy / (hammer['peak_force_n'] / a_sh):.1f} on yield; impact passes through the shoulder, not the thread")
    tau_t = 2 * t_lim / (math.pi * d_p ** 2 * le) * 2
    led.add("bit_joint_thread_shear_mpa", round(tau_t / 1e6, 2), "MPa", "~ 4 T / (pi d_p^2 L_e)",
            "right-hand thread: drilling torque tightens the joint")


def derive_auger(p: Dict[str, Any], led: Ledger, hammer: Dict[str, float], bit: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    a = p["auger_geometry"]
    crit = p["criteria"]
    steel = p["materials"]["auger_steel"]
    cm = p["cutting_model"]
    od = float(a["outer_diameter_m"])
    gauge = float(a["gauge_diameter_m"])
    d_bore = float(a["wiring_bore_m"])
    t_lim = float(p["actuators"]["rotary"]["max_safe_torque_nm"])
    s_u = float(steel["ultimate_strength_pa"])
    s_y = float(steel["yield_strength_pa"])
    s_e = 0.5 * s_u * float(steel["endurance_factor_surface"]) * float(steel["endurance_factor_reliability"])
    led.add("auger_endurance_limit_mpa", round(s_e / 1e6, 1), "MPa", "S_e = 0.5 S_u k_a k_e", "Marin-corrected")
    a_tube = math.pi / 4 * (float(a["tube_outer_diameter_m"]) ** 2 - float(a["tube_inner_diameter_m"]) ** 2)

    # Smallest root OD (0.5 mm steps) passing Goodman fatigue and static yield
    fos_f = float(crit["structural_fos_fatigue"])
    fos_y = float(crit["structural_fos_yield"])
    chosen: Optional[Dict[str, float]] = None
    sweep: List[Dict[str, float]] = []
    d = od - 2 * 0.0005
    while d > d_bore + 0.001:
        a_root = math.pi / 4 * (d ** 2 - d_bore ** 2)
        sigma = hammer["peak_force_n"] * 2.0 / (a_tube + a_root)       # transmitted stress into the root
        tau = 16 * t_lim * d / (math.pi * (d ** 4 - d_bore ** 4))
        sig_a = 0.5 * sigma
        sig_m = math.sqrt((0.5 * sigma) ** 2 + 3 * tau ** 2)
        n_f = 1.0 / (sig_a / s_e + sig_m / s_u)
        n_y = s_y / math.sqrt(sigma ** 2 + 3 * tau ** 2)
        row = {"root_od_mm": round(d * 1e3, 1), "flute_depth_mm": round((od - d) / 2 * 1e3, 2),
               "impact_stress_mpa": round(sigma / 1e6, 1), "torsion_mpa": round(tau / 1e6, 1),
               "fos_fatigue": round(n_f, 2), "fos_yield": round(n_y, 2)}
        sweep.append(row)
        if n_f >= fos_f and n_y >= fos_y:
            chosen = row
        elif chosen is not None:
            break
        d -= 0.0005
    assert chosen is not None
    d_root = chosen["root_od_mm"] / 1e3
    depth = led.add("sampling_flute_depth_mm", chosen["flute_depth_mm"], "mm", "h_f = (D_o - d_root)/2",
                    f"deepest flute whose root passes fatigue FOS {fos_f} and yield FOS {fos_y}")
    led.add("sampling_root_od_mm", chosen["root_od_mm"], "mm", "min d_r : n_Goodman >= FOS",
            f"impact {chosen['impact_stress_mpa']} MPa + torsion {chosen['torsion_mpa']} MPa -> n_f = {chosen['fos_fatigue']}")

    # Sampling pitch: cuttings stay on a stopped flight if tan(alpha_mean) <= mu_flight
    mu_f = float(a["wall_friction_coeff_auger"])
    d_mean = 0.5 * (od + d_root)
    p_samp_raw = mu_f * math.pi * d_mean
    p_samp = led.add("sampling_pitch_mm", math.floor(p_samp_raw * 2e3) / 2, "mm", "p <= mu_f pi d_mean",
                     f"helix angle at mean radius below friction angle atan({mu_f}) = {math.degrees(math.atan(mu_f)):.1f} deg")

    # Sampling length: hold one bite of bulked cuttings x margin
    bite = float(p["system"]["bite_depth_m"])
    bulk = float(cm["cuttings_bulking_factor"])
    a_hole = math.pi / 4 * gauge ** 2
    blockage = float(a["sampling_flight_thickness_m"]) / (p_samp / 1e3)
    a_flute = math.pi / 4 * (od ** 2 - d_root ** 2) * (1 - blockage) + math.pi / 4 * (gauge ** 2 - od ** 2)
    l_need = bite * a_hole * bulk * float(crit["cuttings_retention_margin"]) / a_flute
    l_samp = led.add("sampling_length_mm", math.ceil(l_need * 1e3 / 5) * 5, "mm",
                     "L_s = z_bite A_hole B m / A_flute",
                     f"A_flute = {a_flute*1e6:.0f} mm^2 per mm, bulking {bulk}, margin {crit['cuttings_retention_margin']}")
    z_crit = led.add("choking_onset_bite_m", round(l_samp / 1e3 * a_flute / (a_hole * bulk), 4), "m",
                     "z_crit = L_s A_flute / (A_hole B)", "continuous penetration at which sampling flutes are full")

    # Transport pitch: maximum screw efficiency alpha* = 45 - phi/2 at the transport flute mean diameter
    phi_f = math.degrees(math.atan(mu_f))
    alpha = led.add("transport_helix_angle_deg", round(45.0 - phi_f / 2, 2), "deg", "alpha* = 45 - phi_f/2",
                    "maximises screw efficiency tan(a)/tan(a+phi) (Shigley power-screw result)")
    d_mt = 0.5 * (od + float(a["tube_outer_diameter_m"]))
    p_tr = led.add("transport_pitch_mm", round(math.pi * d_mt * math.tan(math.radians(alpha)) * 1e3, 1), "mm",
                   "p = pi d_mean tan(alpha*)")

    # Centrifugal conveying check at nominal RPM
    rpm = float(p["actuators"]["rotary"]["nominal_rpm"])
    om = rpm * 2 * math.pi / 60
    mu_w = float(a["wall_friction_coeff_soil"])
    conv = {}
    for env_id, env in p["environments"].items():
        g = float(env["gravity_m_s2"])
        ratio = mu_w * om ** 2 * (od / 2) / g
        rpm_min = math.sqrt(g / (mu_w * od / 2)) * 60 / (2 * math.pi)
        conv[env_id] = {"lift_ratio": round(ratio, 3), "rpm_min_conveying": round(rpm_min, 0)}
        led.add(f"conveying_lift_ratio_{env_id}", round(ratio, 3), "-", "mu_w omega^2 r / g",
                f"< 1: cuttings are not lifted continuously -> they stay in the sampling flutes (bite drilling). "
                f"Continuous conveying would need > {rpm_min:.0f} rpm")

    # Wearflat length of the rounded cutter edges (Detournay-Defourny friction term)
    l_wf = led.add("wearflat_total_length_mm",
                   round(float(a["num_cutters"]) * float(bit["edge_radius_m"]) * float(cm["wearflat_length_to_edge_radius"]) * 1e3, 3),
                   "mm", "l = n_c r_edge (l/r)")

    return {"root_od_mm": chosen["root_od_mm"], "flute_depth_mm": depth, "sampling_pitch_mm": p_samp,
            "sampling_length_mm": l_samp, "choking_onset_m": z_crit, "transport_helix_deg": alpha,
            "transport_pitch_mm": p_tr, "flute_area_mm2": round(a_flute * 1e6, 1),
            "hole_area_mm2": round(a_hole * 1e6, 1), "wearflat_mm": l_wf, "conveying": conv, "root_sweep": sweep,
            "tube_area_mm2": round(a_tube * 1e6, 2)}


def derive_thermal_limits(p: Dict[str, Any], led: Ledger, bite_time_s: float) -> Dict[str, float]:
    crit = p["criteria"]
    # Moon: allowed ice recession on a cuttings grain during one bite cycle
    r_g = float(crit["moon_grain_radius_m"])
    frac = float(crit["moon_ice_loss_fraction"])
    recession = r_g * (1 - (1 - frac) ** (1 / 3))
    j_allow = recession / bite_time_s * 917.0
    lo, hi = 110.0, 260.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ice_sublimation_flux(mid) > j_allow:
            hi = mid
        else:
            lo = mid
    t_moon = led.add("critical_bit_temp_moon_k", round(lo, 1), "K",
                     "J_HK(T) = p_v(T) sqrt(M/2piRT) = rho_ice dr/t_bite",
                     f"<= {frac*100:.0f} % ice loss from {r_g*1e6:.0f} um radius grains in {bite_time_s:.0f} s (Murphy & Koop 2005)")
    t_mars = led.add("critical_bit_temp_mars_k", round(273.15 - float(crit["mars_melt_margin_k"]), 2), "K",
                     "T_crit = 273.15 K - margin", "prevents transient melting / refreezing of pore ice on the bit")
    return {"moon": t_moon, "mars": t_mars}


def derive_warmup(p: Dict[str, Any], led: Ledger) -> Dict[str, float]:
    av = p["actuators"]["avionics_and_thermal"]
    crit = p["criteria"]
    c_act = float(av["actuator_thermal_mass_kg"]) * float(av["actuator_specific_heat_j_kgk"])
    t_target = float(crit["lubricant_min_temp_k"]) + float(crit["actuator_warmup_margin_k"])
    led.add("actuator_warmup_target_k", t_target, "K", "T_target = T_lube,min + margin")
    out = {}
    for env_id, env in p["environments"].items():
        t = float(env["initial_surface_temp_k"])
        t_sky = float(env["sky_temp_k"])
        h_conv = float(av["mars_convection_coeff_w_m2k"]) if env_id == "mars" else 0.0
        a_rad = float(av["actuator_radiating_area_m2"])
        eps = float(av["mli_effective_emissivity"])
        t_amb = float(env["initial_surface_temp_k"])
        q_h = float(av["heater_power_w"])
        dt, time = 1.0, 0.0
        while t < t_target and time < 36000:
            loss = eps * SIGMA_SB * a_rad * (t ** 4 - t_sky ** 4) + h_conv * a_rad * (t - t_amb)
            t += (q_h - loss) / c_act * dt
            time += dt
        out[env_id] = led.add(f"warmup_duration_{env_id}_s", int(time), "s",
                              "C dT/dt = Q_h - eps sigma A (T^4 - T_sky^4) - h A (T - T_amb)",
                              f"C = {c_act:.0f} J/K, {env['initial_surface_temp_k']} K -> {t_target} K")
    return {"target_k": t_target, **out}


def derive_bit_thermal_mass(p: Dict[str, Any], led: Ledger, root_od_mm: float) -> float:
    steel = p["materials"]["auger_steel"]
    a = p["auger_geometry"]
    r = float(a["outer_diameter_m"]) / 2
    l_c = float(p["cutting_model"]["bit_conduction_length_m"])
    vol = math.pi * (r - 0.001) ** 2 * 0.022 + math.pi / 4 * ((root_od_mm / 1e3) ** 2 - float(a["wiring_bore_m"]) ** 2) * (l_c - 0.022)
    vol += 2 * 0.010 * 0.003 * 0.008
    c = vol * float(steel["density_kg_m3"]) * float(steel["specific_heat_j_kgk"])
    return led.add("bit_thermal_mass_j_k", round(c, 2), "J/K", "C = rho c_p V (crown + cutters + lowest flutes)")


def derive_all(p: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = p or load_system_params()
    led = Ledger()
    a = p["auger_geometry"]
    a_tube = math.pi / 4 * (float(a["tube_outer_diameter_m"]) ** 2 - float(a["tube_inner_diameter_m"]) ** 2)
    hd = p["drill_head"]
    hammer_mass = (math.pi / 4 * (float(hd["hammer_od_m"]) ** 2 - float(hd["spline_major_m"]) ** 2)
                   * float(hd["hammer_length_m"]) * float(p["materials"]["auger_steel"]["density_kg_m3"]))
    led.add("hammer_mass_kg", round(hammer_mass, 4), "kg", "m_h = rho pi (r_o^2 - r_i^2) h",
            f"hammer ring {hd['hammer_od_m']*1e3:.0f} mm OD on the {hd['spline_major_m']*1e3:.0f} mm spline x {hd['hammer_length_m']*1e3:.0f} mm")
    wob = derive_wob(p, led)
    hammer = derive_hammer(p, led, hammer_mass, a_tube)
    bit = derive_bit(p, led, hammer)
    auger = derive_auger(p, led, hammer, bit)
    finish_bit(p, led, bit, auger, hammer)
    head = derive_head(p, led, hammer, wob)
    string = derive_string(p, led, hammer, auger, head, wob)
    brush = derive_brush(p, led, auger, string)
    auger["brush_sweep_mm_s"] = brush["sweep"]["v_mm_s"]
    auger["transfer_duration_s"] = brush["sweep"]["duration_s"]
    mast = derive_mast(p, led, wob, hammer, head, string, brush)
    # Nominal bite cycle used for the sublimation criterion: drill + retract + brush (~ representative)
    bite_cycle_s = float(p["controller"]["sublimation_exposure_s"])
    t_crit = derive_thermal_limits(p, led, bite_cycle_s)
    warm = derive_warmup(p, led)
    c_bit = derive_bit_thermal_mass(p, led, auger["root_od_mm"])
    rot = p["actuators"]["rotary"]
    led.add("torque_fault_threshold_nm", round(float(rot["max_safe_torque_nm"]) * float(p["criteria"]["torque_fault_fraction"]), 3),
            "N m", "tau_fault = f x tau_max")
    strata = {}
    for env_id, env in p["environments"].items():
        strata[env_id] = [
            {"name": L["name"], "z0": L["z_start_m"], "z1": L["z_end_m"], "ucs_mpa": round(layer_ucs_pa(L) / 1e6, 4),
             "ice_wt_pct": L["ice_wt_pct"], "k_w_mk": L["thermal_conductivity_w_mk"]}
            for L in env["stratigraphy"]
        ]
    return {
        "wob": wob,
        "hammer": hammer,
        "auger": auger,
        "bit": bit,
        "head": head,
        "string": string,
        "brush": brush,
        "mast": mast,
        "critical_temp_k": t_crit,
        "warmup": warm,
        "bit_thermal_mass_j_k": c_bit,
        "torque_fault_nm": led.items["torque_fault_threshold_nm"]["value"],
        "strata": strata,
        "ledger": led.items,
    }


def main() -> None:
    EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    d = derive_all()
    (EXPORTS_DATA_DIR / "design_derivations.json").write_text(json.dumps(d, indent=2))
    print("=== Module 0: First-principles design derivations ===")
    for k, v in d["ledger"].items():
        print(f"  {k:36s} = {v['value']!s:>10} {v['unit']:5s}  [{v['equation']}]")


if __name__ == "__main__":
    main()
