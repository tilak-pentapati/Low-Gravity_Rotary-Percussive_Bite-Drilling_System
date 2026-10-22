"""
Auger-string derivations: transport flight and its weld, the 1.24 m string as a beam (Euler buckling,
lateral and torsional natural frequencies, torsional wind-up), the hex drive shank in the drill-head anvil,
the tube-to-shank braze, the retention pin and the thermocouple path. Called from src/design.py; every
result is written to the derivation ledger.

String model (Euler-Bernoulli / St-Venant finite elements along the string, s = 0 at the bit tip):
  segments   bit crown | sampling root 6/4 mm + flight | bit joint | 14/12 mm tube + transport flight | collar + hex
  flights    add mass (helicoid area x thickness / pitch) but no bending stiffness (an open helix is a coil spring)
  top        clamped where the hex shank enters the anvil socket (52 mm of hex in a 0.1 mm clearance socket)
  supports   bit pinned on the hole bottom; the hole wall and the mast-foot guide bushing are lateral supports
             (radial clearance 0.5 mm: they bound any vibration or buckle to that amplitude)
  buckling   K phi = P K_g phi;  vibration  K phi = omega^2 M phi;  torsion  K_t phi = omega^2 J phi

Flight root (cantilever plate strip, Shigley / Roark):
  plug pressure that would stall the auger at the torque limit, from the terramechanics screw-torque model
      sigma_plug = T_max / (A_tr r_m tan(alpha + phi_f))                       (mean load)
  inertial pressure at each blow's stress-wave front, q_i = DLF rho t a          (alternating, 26 Hz)
  root bending  sigma = 3 q L^2 / t^2   (uniform pressure q on a strip of radial length L, thickness t)
  attachment trade:
    (a) fillet welds: static weld-line shear (Shigley Table 9-2, AWS 0.30 F_EXX) and IIW FAT fatigue of the toe
    (b) integral flight with a root fillet: Goodman with K_f = 1 + q (K_t - 1)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.linalg as sla


def helicoid_area_per_turn(r_i: float, r_o: float, pitch: float) -> float:
    """Area of one turn of a helicoid flight between r_i and r_o: int sqrt((2 pi r)^2 + p^2) dr."""
    r = np.linspace(r_i, r_o, 400)
    return float(np.trapezoid(np.sqrt((2 * math.pi * r) ** 2 + pitch ** 2), r))


def helicoid_polar_per_turn(r_i: float, r_o: float, pitch: float) -> float:
    """int r^2 dA over one turn (polar second moment of the flight surface)."""
    r = np.linspace(r_i, r_o, 400)
    return float(np.trapezoid(r ** 2 * np.sqrt((2 * math.pi * r) ** 2 + pitch ** 2), r))


def string_stations(p: Dict[str, Any], d_auger: Dict[str, Any], stack: Dict[str, float], lap_m: float) -> Dict[str, float]:
    """Axial stations along the string (m from the bit tip). Bit-joint stack mirrors src/cad_bit.py."""
    a, b = p["auger_geometry"], p["drill_bit"]
    L = float(p["cad_layout"]["auger_length_m"])
    crown = float(a["cutter_protrusion_m"]) + float(b["crown_height_m"])
    samp = crown + d_auger["sampling_length_mm"] / 1e3
    tube0 = samp + 0.006 + float(b["joint_engagement_m"]) + 0.004 - 0.0045
    hex0 = L - 0.040                                   # hex shank: 40 mm below to 40 mm above the carriage underside
    return {"crown_top": crown, "sampling_top": samp, "tube_bottom": tube0, "collar_bottom": hex0 - 2 * lap_m,
            "hex_bottom": hex0, "clamp": L + stack["anvil_bottom"] / 1e3, "length": L}


# ------------------------------------------------------------------------------
# Finite-element beam / shaft
# ------------------------------------------------------------------------------
class StringFE:
    """Piecewise-uniform string between the bit tip (s = 0) and the anvil clamp."""

    def __init__(self, segments: Sequence[Tuple[float, float, Dict[str, float]]], clamp: float, n_el: int) -> None:
        self.segs = segments
        self.x = np.linspace(0.0, clamp, n_el + 1)
        self.n_el = n_el

    def _prop(self, s: float) -> Dict[str, float]:
        for s0, s1, pr in self.segs:
            if s0 <= s <= s1:
                return pr
        return self.segs[-1][2]

    def _node(self, s: float) -> int:
        return int(round(s / self.x[-1] * self.n_el))

    def _bending(self, pinned: Sequence[float]):
        n = 2 * len(self.x)
        K, M, G = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for e in range(self.n_el):
            L = self.x[e + 1] - self.x[e]
            pr = self._prop(0.5 * (self.x[e] + self.x[e + 1]))
            ei, m = pr["EI"], pr["m"]
            k = ei / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                        [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
            mm = m * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                         [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
            kg = 1 / (30 * L) * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
                                          [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]])
            ix = np.ix_([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3], [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
            K[ix] += k
            M[ix] += mm
            G[ix] += kg
        fixed = {n - 2, n - 1} | {2 * self._node(s) for s in pinned}
        free = [i for i in range(n) if i not in fixed]
        return K, M, G, free

    def lateral(self, pinned: Sequence[float], axial_n: float = 0.0) -> Dict[str, float]:
        """First lateral natural frequency (under axial compression) and the Euler buckling load."""
        K, M, G, f = self._bending(pinned)
        ix = np.ix_(f, f)
        w2 = sla.eigh(K[ix] - axial_n * G[ix], M[ix], eigvals_only=True)
        pcr = sla.eigh(K[ix], G[ix], eigvals_only=True)
        return {"f1_hz": math.sqrt(max(w2[0], 0.0)) / (2 * math.pi), "p_cr_n": float(pcr[pcr > 0][0])}

    def compliance(self, pinned: Sequence[float], s_load: float, s_out: float) -> float:
        """Lateral deflection at s_out per newton applied at s_load (m/N)."""
        K, _, _, f = self._bending(pinned)
        fv = np.zeros(K.shape[0])
        fv[2 * self._node(s_load)] = 1.0
        u = np.zeros(K.shape[0])
        u[f] = np.linalg.solve(K[np.ix_(f, f)], fv[f])
        return float(u[2 * self._node(s_out)])

    def torsion(self) -> Dict[str, float]:
        """Clamped at the anvil, bit free: first torsional frequency; twist = T int ds / GJ."""
        n = len(self.x)
        K, J = np.zeros((n, n)), np.zeros((n, n))
        flex = 0.0
        for e in range(self.n_el):
            L = self.x[e + 1] - self.x[e]
            pr = self._prop(0.5 * (self.x[e] + self.x[e + 1]))
            ix = np.ix_([e, e + 1], [e, e + 1])
            K[ix] += pr["GJ"] / L * np.array([[1, -1], [-1, 1]])
            J[ix] += pr["rhoJ"] * L / 6 * np.array([[2, 1], [1, 2]])
            flex += L / pr["GJ"]
        f = list(range(n - 1))
        w2 = sla.eigh(K[np.ix_(f, f)], J[np.ix_(f, f)], eigvals_only=True)
        return {"f1_hz": math.sqrt(w2[0]) / (2 * math.pi), "flexibility_rad_per_nm": flex}


# ------------------------------------------------------------------------------
# Derivation
# ------------------------------------------------------------------------------
def derive_string(p: Dict[str, Any], led, hammer: Dict[str, Any], auger: Dict[str, Any], head: Dict[str, Any],
                  wob: Dict[str, Any]) -> Dict[str, Any]:
    a, s_cfg = p["auger_geometry"], p["auger_string"]
    steel = p["materials"]["auger_steel"]
    e_mod, rho, nu = float(steel["youngs_modulus_pa"]), float(steel["density_kg_m3"]), float(steel["poisson"])
    g_mod = e_mod / (2 * (1 + nu))
    s_u, s_y = float(steel["ultimate_strength_pa"]), float(steel["yield_strength_pa"])
    cr = p["criteria"]
    t_max = float(p["actuators"]["rotary"]["max_safe_torque_nm"])
    f_blow = float(hammer["blow_frequency_hz"])
    r_o = float(a["outer_diameter_m"]) / 2
    r_to, r_ti = float(a["tube_outer_diameter_m"]) / 2, float(a["tube_inner_diameter_m"]) / 2
    r_root, r_bore = auger["root_od_mm"] / 2e3, float(a["wiring_bore_m"]) / 2
    t_tr, t_sa = float(a["transport_flight_thickness_m"]), float(a["sampling_flight_thickness_m"])
    p_tr, p_sa = auger["transport_pitch_mm"] / 1e3, auger["sampling_pitch_mm"] / 1e3
    stack = head["stack_mm"]
    out: Dict[str, Any] = {}

    # ---- Transport flight root under the stalling plug pressure ------------------------------------
    mu_f = float(a["wall_friction_coeff_auger"])
    alpha = math.radians(auger["transport_helix_deg"])
    a_tr = math.pi * (r_o ** 2 - r_to ** 2)
    r_m = 0.5 * (r_o + r_to)
    q = t_max / (a_tr * r_m * math.tan(alpha + math.atan(mu_f)))
    led.add("flight_stall_plug_pressure_kpa", round(q / 1e3, 1), "kPa", "sigma = T_max / (A_tr r_m tan(alpha + phi_f))",
            "largest plug pressure the motor can push against before it stalls (screw-torque model)")
    l_fl = r_o - r_to
    sig_root = 3 * q * l_fl ** 2 / t_tr ** 2
    led.add("flight_root_bending_mpa", round(sig_root / 1e6, 1), "MPa", "sigma = 3 q L^2 / t^2",
            f"flight strip L = {l_fl*1e3:.1f} mm, t = {t_tr*1e3:.1f} mm; yield FOS {s_y/sig_root:.1f}")
    # Inertial load at the stress-wave front: particle velocity v = sigma/(rho c) reached in ~ 2 L_h / c,
    # applied suddenly (dynamic load factor) -> q_i = DLF rho t a, once per blow
    c0 = math.sqrt(e_mod / rho)
    a_tube = math.pi * (r_to ** 2 - r_ti ** 2)
    sig_wave = hammer["peak_force_n"] / a_tube
    acc = (sig_wave / (rho * c0)) / (2 * float(p["drill_head"]["hammer_length_m"]) / c0)
    q_i = float(s_cfg["impact_load_amplification"]) * rho * t_tr * acc
    sig_inert = 3 * q_i * l_fl ** 2 / t_tr ** 2
    led.add("flight_impact_inertia_mpa", round(sig_inert / 1e6, 1), "MPa", "q = DLF rho t a,  a = (sigma/rho c) / (2 L_h / c)",
            f"wave-front acceleration {acc/9.81:,.0f} g, DLF {s_cfg['impact_load_amplification']}: cycles at the blow rate")
    n_req = f_blow * float(p["drill_head"]["design_operating_hours"]) * 3600 * float(cr["mechanism_life_factor"])
    led.add("flight_required_cycles", float(f"{n_req:.3g}"), "-", "N = f_blow x t_drill x life factor")
    # (a) Fillet welds both sides: static shear (AWS) and IIW fatigue of the weld toe
    w_leg = float(s_cfg["flight_weld_leg_m"])
    throat = 0.707 * w_leg
    q_tot = q + q_i
    d_c = t_tr + 2 * w_leg / 3
    tau_w = math.hypot(q_tot * l_fl ** 2 / 2 / (throat * d_c), q_tot * l_fl / (2 * throat))
    tau_allow = float(s_cfg["weld_allowable_ratio"]) * s_u
    led.add("flight_weld_shear_mpa", round(tau_w / 1e6, 1), "MPa", "tau = sqrt((M'/(a d_c))^2 + (V'/2a)^2), a = 0.707 w",
            f"static: {w_leg*1e3:.1f} mm fillets, allowable 0.30 F_EXX = {tau_allow/1e6:.0f} MPa (ratio {tau_allow/tau_w:.1f})")
    fat = float(s_cfg["weld_fat_class_pa"])
    n_knee = float(s_cfg["weld_cafl_cycles"])
    cafl = fat * (2e6 / n_knee) ** (1 / 3)
    life_w = 2e6 * (fat / sig_inert) ** 3 if sig_inert > cafl else float("inf")
    weld_ok = tau_w <= tau_allow and sig_inert <= cafl
    led.add("flight_weld_fatigue_life_cycles", float(f"{life_w:.3g}"), "-", "N = 2e6 (FAT / dsigma)^3 above the CAFL",
            f"stress range {sig_inert/1e6:.0f} MPa vs constant-amplitude limit {cafl/1e6:.0f} MPa (IIW FAT {fat/1e6:.0f}): "
            + ("passes" if weld_ok else f"fails the {n_req:.2g}-cycle requirement -> welded flights rejected"))
    # (b) Flight machined integral with the tube, root fillet r = t/2: Goodman with the Peterson notch factor
    kf = 1 + float(s_cfg["notch_sensitivity"]) * (float(s_cfg["flight_root_kt"]) - 1)
    s_e = 0.5 * s_u * float(steel["endurance_factor_surface"]) * float(steel["endurance_factor_reliability"])
    sig_a, sig_m = kf * sig_inert / 2, kf * (sig_root + sig_inert / 2)
    n_int = 1.0 / (sig_a / s_e + sig_m / s_u)
    led.add("flight_root_fatigue_fos", round(n_int, 2), "-", "1 / (K_f sigma_a / S_e + K_f sigma_m / S_u)",
            f"integral flight, root fillet r = {s_cfg['flight_root_fillet_to_t']} t, K_f = {kf:.2f}; plug load as mean, blows as alternating")
    integral = n_int >= float(cr["structural_fos_fatigue"])
    assert integral or weld_ok, "no flight attachment passes"
    choice = "welded" if weld_ok else "integral"
    out["flight"] = {"plug_pa": q, "root_mpa": sig_root / 1e6, "inertia_mpa": sig_inert / 1e6, "weld_mpa": tau_w / 1e6,
                     "weld_allow_mpa": tau_allow / 1e6, "weld_life": life_w, "cafl_mpa": cafl / 1e6, "required_cycles": n_req,
                     "integral_fos": n_int, "kf": kf, "attachment": choice, "weld_leg_m": w_leg,
                     "root_fillet_m": float(s_cfg["flight_root_fillet_to_t"]) * t_tr}

    # ---- Tube-to-shank braze and hex shank -----------------------------------------------------------
    t_wall = r_to - r_ti
    lap = float(s_cfg["braze_lap_to_wall"]) * t_wall
    tau_b_allow = float(p["drill_bit"]["braze_shear_strength_pa"])
    l_need = 2 * t_max * float(cr["structural_fos_yield"]) / (math.pi * (2 * r_to) ** 2 * tau_b_allow)
    lap = max(lap, math.ceil(l_need / 0.0005) * 0.0005)
    led.add("shank_braze_lap_mm", round(lap * 1e3, 2), "mm", "L_b = max(3 t, 2 T FOS / (pi d^2 tau_b))",
            f"torque alone needs {l_need*1e3:.2f} mm; the 3 t lap rule (AWS) governs")
    pull = float(head["load_cell_n"])
    tau_bt = 2 * t_max / (math.pi * (2 * r_to) ** 2 * lap)
    tau_bp = pull / (math.pi * 2 * r_to * lap)
    led.add("shank_braze_fos", round(tau_b_allow / math.hypot(tau_bt, tau_bp), 1), "-", "tau_b / sqrt(tau_T^2 + tau_pull^2)",
            f"torque {tau_bt/1e6:.1f} MPa + pull-out at the {pull:.0f} N load-cell rating {tau_bp/1e6:.1f} MPa; impact bears on the socket floor")

    st = string_stations(p, auger, stack, lap)
    hex_r = float(s_cfg["hex_across_corners_m"]) / 2
    s_af = 2 * hex_r * math.cos(math.radians(30))
    r_sb = float(p["drill_head"]["spindle_bore_m"]) / 2
    tau_hex = 16 * t_max * s_af / (math.pi * (s_af ** 4 - (2 * r_sb) ** 4))
    led.add("hex_shank_torsion_mpa", round(tau_hex / 1e6, 1), "MPa", "tau = 16 T s / (pi (s^4 - d_b^4))",
            f"inscribed circle of the {s_af*1e3:.2f} mm A/F hex (conservative vs Roark's hexagon); bore {2*r_sb*1e3:.0f} mm")
    engage = (stack["hex_top"] - stack["anvil_bottom"]) / 1e3
    h_flank = hex_r - s_af / 2
    d_mf = hex_r + s_af / 2
    p_flank = 2 * t_max / (d_mf * h_flank * engage * 6 * float(p["drill_head"]["spline_load_sharing"]))
    led.add("hex_socket_flank_pressure_mpa", round(p_flank / 1e6, 2), "MPa", "p = 2 T / (d_m h L n k)",
            f"6 flats, {engage*1e3:.0f} mm in the anvil socket; allowable {float(p['drill_head']['spline_allowable_pressure_pa'])/1e6:.0f} MPa (as the spline)")
    d_pin = float(s_cfg["retention_pin_d_m"])
    a_hex = 1.5 * math.sqrt(3) * hex_r ** 2 - math.pi * r_sb ** 2
    a_net = a_hex - d_pin * (s_af - 2 * r_sb)
    sig_hex = hammer["peak_force_n"] / a_net
    led.add("hex_shank_impact_stress_mpa", round(sig_hex / 1e6, 1), "MPa", "F_imp / A_net (pin section)",
            f"net hex area {a_net*1e6:.0f} mm^2; yield FOS {s_y/sig_hex:.1f}")
    tau_pin = pull / (2 * math.pi / 4 * d_pin ** 2)
    led.add("retention_pin_shear_mpa", round(tau_pin / 1e6, 1), "MPa", "tau = F / (2 pi d^2/4)",
            f"double shear at the {pull:.0f} N load-cell rating (the largest pull the feed can apply and still measure)")
    led.add("hex_shank_engagement_mm", round(engage * 1e3, 1), "mm", "hex_top - anvil_bottom (drill-head stack)")
    out["shank"] = {"lap_m": lap, "across_flats_m": s_af, "hex_r_m": hex_r, "torsion_mpa": tau_hex / 1e6,
                    "flank_mpa": p_flank / 1e6, "impact_mpa": sig_hex / 1e6, "pin_mpa": tau_pin / 1e6, "pin_d_m": d_pin,
                    "engagement_m": engage, "pull_n": pull,
                    "pin_z_mm": stack["anvil_bottom"] + 2 * d_pin * 1e3}     # 2 d edge distance from the anvil nose face

    # ---- String beam model ---------------------------------------------------------------------------
    def tube_props(ro: float, ri: float, extra_area: float = 0.0, extra_polar: float = 0.0) -> Dict[str, float]:
        area = math.pi * (ro ** 2 - ri ** 2)
        i = math.pi / 4 * (ro ** 4 - ri ** 4)
        return {"EI": e_mod * i, "GJ": g_mod * 2 * i, "m": rho * (area + extra_area), "rhoJ": rho * (2 * i + extra_polar), "c": ro}

    fl_s = t_sa / p_sa
    fl_t = t_tr / p_tr
    segs = [
        (0.0, st["crown_top"], tube_props(r_o - 0.0005, r_bore)),
        (st["crown_top"], st["sampling_top"], tube_props(r_root, r_bore, fl_s * helicoid_area_per_turn(r_root, r_o, p_sa),
                                                         fl_s * helicoid_polar_per_turn(r_root, r_o, p_sa))),
        (st["sampling_top"], st["tube_bottom"], tube_props(float(p["drill_bit"]["joint_shoulder_od_m"]) / 2, r_bore)),
        (st["tube_bottom"], st["collar_bottom"], tube_props(r_to, r_ti, fl_t * helicoid_area_per_turn(r_to, r_o, p_tr),
                                                            fl_t * helicoid_polar_per_turn(r_to, r_o, p_tr))),
        (st["collar_bottom"], st["clamp"] + 1e-9, tube_props(s_af / 2, r_sb)),
    ]
    fe = StringFE(segs, st["clamp"], int(s_cfg["fe_elements"]))
    mass = sum((s1 - s0) * pr["m"] for s0, s1, pr in segs)
    led.add("auger_string_mass_kg", round(mass, 3), "kg", "sum rho (A_tube + A_flight t/p) ds",
            "bit to anvil; flights add mass, not bending stiffness")

    foot = float(p["cad_layout"]["foot_top_m"])
    l_bush = float(s_cfg["guide_bushing_length_m"])
    s_bush0 = foot - 0.020                                   # bushing sits in the foot-plate boss, 20 mm below its top
    p_design = max(w["wob_limit_n"] for w in wob.values())   # the lander lifts off before the string sees more
    sep = float(s_cfg["resonance_separation"])
    target = float(p["system"]["target_depth_m"])
    sweep: List[Dict[str, float]] = []
    for z in np.linspace(0.0, target, 11):
        pins = [0.0] + [float(s) for s in fe.x if s <= z] + [z + s_bush0, z + s_bush0 + l_bush]
        lat = fe.lateral(pins, p_design)
        sweep.append({"depth_m": round(float(z), 2), "f1_hz": round(lat["f1_hz"], 1), "p_cr_n": round(lat["p_cr_n"], 0)})
    worst_f = min(sweep, key=lambda r: r["f1_hz"])
    worst_p = min(sweep, key=lambda r: r["p_cr_n"])
    free = fe.lateral([0.0], p_design)
    led.add("string_buckling_load_n", worst_p["p_cr_n"], "N", "K phi = P K_g phi (bushing + hole wall as supports)",
            f"lowest over the stroke at {worst_p['depth_m']:.1f} m; FOS {worst_p['p_cr_n']/p_design:.1f} on the {p_design:.0f} N lift-off WOB")
    led.add("string_buckling_no_bushing_n", round(free["p_cr_n"], 0), "N", "clamped at the anvil, pinned at the bit only",
            f"FOS {free['p_cr_n']/p_design:.1f}: still stable, the bushing is not needed for buckling")
    led.add("string_lateral_f1_hz", worst_f["f1_hz"], "Hz", "K phi = omega^2 M phi, under the lift-off WOB",
            f"lowest over the stroke (start of the hole, longest free span); {worst_f['f1_hz']/f_blow:.2f} x the {f_blow:.0f} Hz blow rate")
    led.add("string_lateral_no_bushing_hz", round(free["f1_hz"], 1), "Hz", "same model without the mast-foot guide bushing",
            f"{free['f1_hz']/f_blow:.2f} x blow rate: inside the +/-{sep*100:.0f} % band, so the bushing is required")
    rpm_max = float(p["actuators"]["rotary"]["max_rpm"])
    led.add("string_whirl_margin", round(worst_f["f1_hz"] / (rpm_max / 60), 1), "-", "f1 / (N_max / 60)",
            "first lateral mode vs the fastest shaft speed (no whirl critical speed in the RPM range)")
    tor = fe.torsion()
    led.add("string_torsional_f1_hz", round(tor["f1_hz"], 0), "Hz", "K_t phi = omega^2 J phi (clamped at the anvil, bit free)",
            f"{tor['f1_hz']/f_blow:.1f} x the blow rate")
    twist = t_max * tor["flexibility_rad_per_nm"]
    led.add("string_windup_deg", round(math.degrees(twist), 2), "deg", "theta = T int ds / GJ",
            f"at the {t_max} N m limit; the 6/4 mm sampling root is {100*(st['sampling_top']-st['crown_top'])/(g_mod*math.pi/2*(r_root**4-r_bore**4))/tor['flexibility_rad_per_nm']:.0f} % of the compliance")
    led.add("string_windup_energy_j", round(0.5 * t_max * twist, 3), "J", "U = T theta / 2",
            "released into the bit if it breaks free after a stall (stick-slip)")
    tau_tube = t_max * r_to / (math.pi / 2 * (r_to ** 4 - r_ti ** 4))
    led.add("transport_tube_torsion_mpa", round(tau_tube / 1e6, 1), "MPa", "tau = T r_o / J")

    # Stiffness the brush pushes against while the bit is above the bushing (TRANSFER / re-entry pose)
    brush_h = float(p["cad_layout"]["brush_axle_height_m"])
    s_reentry = s_bush0 + l_bush                                # bit tip just above the bushing
    s_brush = brush_h - s_reentry
    tip_per_n = abs(fe.compliance([], s_brush, 0.0))
    out["beam"] = {"sweep": sweep, "f1_hz": worst_f["f1_hz"], "f1_no_bushing_hz": free["f1_hz"], "p_cr_n": worst_p["p_cr_n"],
                   "p_cr_no_bushing_n": free["p_cr_n"], "p_design_n": p_design, "torsion_f1_hz": tor["f1_hz"],
                   "windup_rad": twist, "mass_kg": mass, "tip_compliance_m_per_n": tip_per_n, "separation": sep,
                   "tube_torsion_mpa": tau_tube / 1e6}
    out["stations_m"] = st
    out["segments"] = [[s0, s1, pr] for s0, s1, pr in segs]   # beam model reused for the stowed (launch) string, src/design_mast.py
    out["bushing"] = {"bore_m": float(a["gauge_diameter_m"]) + 2 * float(s_cfg["guide_bushing_radial_clearance_m"]),
                      "length_m": l_bush, "z0_m": s_bush0, "leadin_m": float(s_cfg["guide_bushing_leadin_m"])}

    # ---- Thermocouple path ------------------------------------------------------------------------
    lead = float(s_cfg["thermocouple_lead_od_m"])
    bores = {"sampling root": 2 * r_bore, "transport tube": 2 * r_ti, "hex shank / spindle": 2 * r_sb}
    narrow = min(bores, key=bores.get)
    led.add("thermocouple_path_fill", round((lead / bores[narrow]) ** 2, 3), "-", "(d_lead / d_bore)^2",
            f"narrowest bore: {narrow} {bores[narrow]*1e3:.0f} mm; lead {lead*1e3:.1f} mm runs bit -> tube -> spindle -> slip ring")
    out["lead_od_m"] = lead
    return out
