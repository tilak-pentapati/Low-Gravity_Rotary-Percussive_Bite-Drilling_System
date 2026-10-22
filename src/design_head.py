"""
Drill-head derivations: hammer spring, rotary drive (BLDC + 8:1 planetary + 24T/12T spur), barrel cam,
percussive drive, spindle spline, catalogue bearing lives, WOB load cell, leadscrew nut and the axial
stack of the enclosed head. Called from src/design.py; every result is written to the derivation ledger.

Sources of the tabulated constants:
  Lewis form factor Y (20 deg full depth)     Shigley's MED, Table 14-2
  Barth velocity factor (cut profile)         Shigley Eq. 14-4b, K_v = (6.1 + V) / 6.1
  AGMA geometry factor I (spur, external)     I = cos(phi) sin(phi) / 2 x m_G / (m_G +/- 1)
  Deep-groove axial factors e, Y vs f0 Fa/C0  SKF Rolling bearings catalogue (normal clearance)
  Paired 25 deg angular-contact factors       SKF catalogue (7xxx ACD, back-to-back pair)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

LEWIS_Y = [(12, 0.245), (13, 0.261), (14, 0.277), (15, 0.290), (16, 0.296), (17, 0.303), (18, 0.309),
           (19, 0.314), (20, 0.322), (21, 0.328), (22, 0.331), (24, 0.337), (26, 0.346), (28, 0.353),
           (30, 0.359), (34, 0.371), (38, 0.384), (43, 0.397), (50, 0.409), (60, 0.422), (75, 0.435),
           (100, 0.447), (150, 0.460), (300, 0.472), (400, 0.480), (10000, 0.485)]
DGBB_AXIAL = [(0.172, 0.19, 2.30), (0.345, 0.22, 1.99), (0.689, 0.26, 1.71), (1.03, 0.28, 1.55),
              (1.38, 0.30, 1.45), (2.07, 0.34, 1.31), (3.45, 0.38, 1.15), (5.17, 0.42, 1.04), (6.89, 0.44, 1.00)]


def _interp(table, x, col=1):
    if x <= table[0][0]:
        return table[0][col]
    for a, b in zip(table, table[1:]):
        if x <= b[0]:
            return a[col] + (b[col] - a[col]) * (x - a[0]) / (b[0] - a[0])
    return table[-1][col]


def lewis_y(z: int) -> float:
    return _interp(LEWIS_Y, z)


def _l10h(c: float, p: float, rpm: float) -> float:
    return (c / p) ** 3 * 1e6 / (60.0 * rpm)


# ------------------------------------------------------------------------------
# Hammer spring
# ------------------------------------------------------------------------------
def derive_spring(p: Dict[str, Any], led, k: float, h: float, f_blow: float) -> Dict[str, Any]:
    """
    Smallest stock wire that meets every spring criterion. For each wire d the coil diameter follows from the
    rate with the minimum active-coil count, D = (G d^4 / (8 k n_a))^(1/3), rounded down to 0.5 mm.
    Checks: Wahl static stress, Zimmerli-Goodman fatigue (0 -> F_max), buckling (L0/D < 2.63/alpha, alpha 0.5),
    coil surge f = 1/2 sqrt(k/m_coils) >= ratio x blow rate. Installed at free length: E = 1/2 k h^2.
    """
    w = p["materials"]["hammer_spring_wire"]
    cr = p["criteria"]
    hd = p["drill_head"]
    g_mod, rho = float(w["shear_modulus_pa"]), float(w["density_kg_m3"])
    s_su = 0.67 * float(w["ultimate_tensile_pa"])
    ssa, ssm = float(w["zimmerli_ssa_pa"]), float(w["zimmerli_ssm_pa"])
    s_se = ssa / (1.0 - ssm / s_su)
    s_sa_allow = s_se * s_su / (s_su + s_se)          # Goodman, load line r = tau_a / tau_m = 1
    f_max = k * h
    n_min = float(cr["spring_min_active_coils"])
    xi = float(cr["spring_clash_allowance"])
    rows: List[Dict[str, Any]] = []
    chosen = None
    for d in hd["spring_wire_options_m"]:
        d = float(d)
        dc = math.floor((g_mod * d ** 4 / (8 * k * n_min)) ** (1 / 3) / 0.0005) * 0.0005
        n_a = g_mod * d ** 4 / (8 * dc ** 3 * k)
        c = dc / d
        kw = (4 * c - 1) / (4 * c - 4) + 0.615 / c
        kb = (4 * c + 2) / (4 * c - 3)
        tau = kw * 8 * f_max * dc / (math.pi * d ** 3)
        tau_a = kb * 8 * (f_max / 2) * dc / (math.pi * d ** 3)
        n_f = s_sa_allow / tau_a
        l_solid = (n_a + 2) * d                          # squared and ground ends
        l_free = l_solid + (1 + xi) * h
        m_coil = rho * math.pi ** 2 * d ** 2 * dc * n_a / 4
        f_surge = 0.5 * math.sqrt(k / m_coil)
        row = {"wire_mm": round(d * 1e3, 2), "coil_d_mm": round(dc * 1e3, 1), "index": round(c, 2),
               "active_coils": round(n_a, 2), "tau_static_mpa": round(tau / 1e6, 1), "fos_fatigue": round(n_f, 2),
               "slenderness": round(l_free / dc, 2), "surge_hz": round(f_surge, 0),
               "free_length_mm": round(l_free * 1e3, 1), "solid_length_mm": round(l_solid * 1e3, 1)}
        row["ok"] = bool(4 <= c <= 12 and tau <= float(w["allowable_shear_pa"])
                         and n_f >= float(cr["structural_fos_fatigue"]) and l_free / dc < 2.63 / 0.5
                         and f_surge >= float(cr["spring_surge_ratio"]) * f_blow)
        rows.append(row)
        if row["ok"] and chosen is None:
            chosen = row
    assert chosen is not None, "no stock wire meets the spring criteria"
    s = chosen
    led.add("hammer_spring_wire_mm", s["wire_mm"], "mm", "min d : all spring checks pass",
            f"17-7PH; coil D = (G d^4 / 8 k n_a)^(1/3) = {s['coil_d_mm']} mm, index {s['index']}")
    led.add("hammer_spring_active_coils", s["active_coils"], "-", "n_a = G d^4 / (8 D^3 k)",
            f">= {n_min:.0f} for a linear rate; squared & ground ends")
    led.add("hammer_spring_free_length_mm", s["free_length_mm"], "mm", "L0 = (n_a + 2) d + (1 + xi) h",
            f"solid {s['solid_length_mm']} mm, clash allowance xi = {xi}")
    led.add("hammer_spring_shear_mpa", s["tau_static_mpa"], "MPa", "tau = K_w 8 F D / (pi d^3)",
            f"F = k h = {f_max:.0f} N; allowable {float(w['allowable_shear_pa'])/1e6:.0f} MPa")
    led.add("hammer_spring_fatigue_fos", s["fos_fatigue"], "-", "n = S_sa / tau_a (Zimmerli-Goodman, r = 1)",
            f"unpeened Zimmerli data, S_su = 0.67 S_ut; required {cr['structural_fos_fatigue']}")
    led.add("hammer_spring_slenderness", s["slenderness"], "-", "L0 / D < 2.63 / alpha",
            "alpha = 0.5 (ends on parallel plates): absolutely stable, no buckling")
    led.add("hammer_spring_surge_hz", s["surge_hz"], "Hz", "f = 1/2 sqrt(k / m_coils)",
            f">= {cr['spring_surge_ratio']:.0f} x {f_blow:.0f} Hz blow rate")
    return {**s, "rows": rows, "f_max_n": f_max}


# ------------------------------------------------------------------------------
# Gears
# ------------------------------------------------------------------------------
def _mesh(p, z1: int, z2: int, m: float, ft: float, v: float, internal: bool = False) -> Dict[str, float]:
    """Lewis bending (both members) and AGMA contact stress for one mesh at module m (SI)."""
    gs = p["materials"]["gear_steel"]
    phi = math.radians(float(p["drill_head"]["pressure_angle_deg"]))
    b = float(p["drill_head"]["face_width_to_module"]) * m
    kv = (6.1 + v) / 6.1
    y1 = lewis_y(z1)
    y2 = 0.485 if internal else lewis_y(z2)              # internal teeth: thicker roots than a rack (upper bound)
    sb = kv * ft / (b * m * min(y1, y2))
    mg = z2 / z1
    i_f = math.cos(phi) * math.sin(phi) / 2 * mg / (mg - 1 if internal else mg + 1)
    sc = float(gs["elastic_coeff_sqrt_pa"]) * math.sqrt(kv * ft / (b * m * z1 * i_f))
    return {"module_mm": m * 1e3, "face_mm": round(b * 1e3, 2), "ft_n": round(ft, 1), "kv": round(kv, 3),
            "bending_mpa": round(sb / 1e6, 1), "contact_mpa": round(sc / 1e6, 1),
            "fos_bending": round(float(gs["bending_strength_pa"]) / sb, 2),
            "fos_contact": round(float(gs["contact_strength_pa"]) / sc, 2)}


def _pick_module(p, check, min_m: float = 0.0) -> Dict[str, Any]:
    """Smallest standard module whose meshes (check(m) -> list of mesh dicts) pass both AGMA factors."""
    cr = p["criteria"]
    for m_mm in p["drill_head"]["standard_modules_mm"]:
        m = float(m_mm) / 1e3
        if m < min_m - 1e-12:
            continue
        meshes = check(m)
        if all(x["fos_bending"] >= cr["gear_bending_fos"] and x["fos_contact"] >= cr["gear_contact_fos"] for x in meshes):
            return {"m": m, "meshes": meshes}
    raise AssertionError("no standard module passes")


# ------------------------------------------------------------------------------
# Drill head
# ------------------------------------------------------------------------------
def derive_head(p: Dict[str, Any], led, hammer: Dict[str, Any], wob: Dict[str, Any]) -> Dict[str, Any]:
    hd = p["drill_head"]
    cr = p["criteria"]
    rot = p["actuators"]["rotary"]
    perc = p["actuators"]["percussive"]
    zf = p["actuators"]["z_axis_feed"]
    out: Dict[str, Any] = {}

    # ---- Rotary drive train ---------------------------------------------------------------------
    t_out = float(rot["max_safe_torque_nm"])
    ratio = float(rot["gear_ratio"])
    z_p, z_g = int(hd["spur_pinion_teeth"]), int(hd["spur_gear_teeth"])
    i_spur = z_g / z_p
    i_pl = ratio / i_spur
    eta_sp = float(hd["spur_efficiency"])
    eta_pl = float(rot["gear_efficiency"]) / eta_sp
    z_s = int(hd["planetary_sun_teeth"])
    z_r = int(round((i_pl - 1) * z_s))
    z_pl = (z_r - z_s) // 2
    n_p = int(hd["planet_count"])
    assert abs(1 + z_r / z_s - i_pl) < 1e-9 and (z_r - z_s) % 2 == 0
    assemble = (z_s + z_r) % n_p == 0
    neighbour = (z_s + z_pl) * math.sin(math.pi / n_p) > z_pl + 2
    led.add("planetary_teeth", f"{z_s}/{z_pl}/{z_r}", "sun/planet/ring", "i = 1 + Z_r/Z_s, Z_p = (Z_r - Z_s)/2",
            f"{n_p} planets: assembly (Z_s+Z_r)/n integer {assemble}, neighbour clearance {neighbour}")

    t_motor = t_out / (ratio * float(rot["gear_efficiency"]))
    n_motor_max = float(rot["max_rpm"]) * ratio
    trv = float(hd["motor_trv_n_m_m3"])
    lam = float(hd["motor_rotor_aspect"])
    d_rot = (4 * t_motor / (math.pi * trv * lam)) ** (1 / 3)
    d_stat = d_rot / float(hd["motor_split_ratio"])
    p_shaft = t_motor * n_motor_max * 2 * math.pi / 60
    led.add("rotary_motor_torque_nm", round(t_motor, 3), "N m", "T_m = T_max / (i eta_g)",
            f"{t_out} N m at the spindle through {ratio:.0f}:1, eta = {rot['gear_efficiency']}")
    led.add("rotary_motor_rotor_d_mm", round(d_rot * 1e3, 1), "mm", "D_r = (4 T / (pi TRV lambda))^(1/3)",
            f"TRV = {trv/1e3:.0f} kN m/m^3, L/D = {lam}; stack {lam*d_rot*1e3:.1f} mm")
    led.add("rotary_motor_stator_od_mm", round(d_stat * 1e3, 1), "mm", "D_s = D_r / split ratio",
            f"frameless BLDC kit class; {n_motor_max:.0f} rpm max, {p_shaft:.0f} W shaft power at the torque limit")

    # Planetary (common module for sun/planet/ring)
    t_sun = t_motor
    n_rel = n_motor_max * (1 - 1 / i_pl)                 # sun speed relative to the carrier
    kg = float(hd["planet_load_sharing"])

    def pl_check(m):
        ft = kg * t_sun / (n_p * m * z_s / 2)
        v = math.pi * m * z_s * n_rel / 60
        return [_mesh(p, z_s, z_pl, m, ft, v), _mesh(p, z_pl, z_r, m, ft, v, internal=True)]

    pl = _pick_module(p, pl_check)
    m_pl = pl["m"]
    led.add("planetary_module_mm", m_pl * 1e3, "mm", "min m : sigma_b <= S_t/S_F and sigma_H <= S_c/S_H",
            f"sun mesh F_t = K_g T_s / (n r_s); bending FOS {pl['meshes'][0]['fos_bending']}, contact FOS {pl['meshes'][0]['fos_contact']}")

    t_pin = t_out / i_spur / eta_sp
    n_pin = float(rot["max_rpm"]) * i_spur

    def sp_check(m):
        return [_mesh(p, z_p, z_g, m, t_pin / (m * z_p / 2), math.pi * m * z_p * n_pin / 60)]

    sp = _pick_module(p, sp_check)
    m_sp = sp["m"]
    a_sp = m_sp * (z_p + z_g) / 2
    led.add("spur_module_mm", m_sp * 1e3, "mm", "Lewis: sigma = K_v F_t / (b m Y);  AGMA: sigma_H = Z_E sqrt(K_v F_t / (b d I))",
            f"{z_p}T/{z_g}T, centre distance {a_sp*1e3:.1f} mm; bending FOS {sp['meshes'][0]['fos_bending']}, contact FOS {sp['meshes'][0]['fos_contact']}")
    led.add("spur_bending_mpa", sp["meshes"][0]["bending_mpa"], "MPa", "sigma_b = K_v F_t / (b m Y)",
            f"S_t = {p['materials']['gear_steel']['bending_strength_pa']/1e6:.0f} MPa (9310 carburised)")
    led.add("spur_contact_mpa", sp["meshes"][0]["contact_mpa"], "MPa", "sigma_H = Z_E sqrt(K_v F_t / (b d_p I))",
            f"S_c = {p['materials']['gear_steel']['contact_strength_pa']/1e6:.0f} MPa")
    out["rotary"] = {"motor_torque_nm": t_motor, "rotor_d_m": d_rot, "rotor_l_m": lam * d_rot, "stator_od_m": d_stat,
                     "motor_rpm_max": n_motor_max, "shaft_power_w": p_shaft, "planetary_teeth": [z_s, z_pl, z_r],
                     "planets": n_p, "planetary_module_m": m_pl, "planetary": pl["meshes"], "spur_module_m": m_sp,
                     "spur": sp["meshes"][0], "spur_teeth": [z_p, z_g], "centre_distance_m": a_sp}

    # ---- Barrel cam (cycloidal rise) ------------------------------------------------------------
    k = float(perc["spring_rate_n_m"])
    h = float(perc["cam_lift_m"])
    lobes = int(perc["cam_lobes"])
    cam_rps = float(perc["cam_speed_rps"])                 # relative to the hammer (which turns with the spindle)
    m_h = float(hammer["mass_kg"])
    gap = float(hd["cam_rise_gap_m"])
    h_c = h + gap
    t_fall = 0.5 * math.pi * math.sqrt(m_h / k)            # spring-driven quarter period
    dwell = math.ceil(float(cr["hammer_fall_time_margin"]) * 360.0 * cam_rps * t_fall / 5.0) * 5.0
    beta = 360.0 / lobes - dwell
    led.add("hammer_fall_time_ms", round(t_fall * 1e3, 2), "ms", "t_f = (pi/2) sqrt(m_h / k)", "hammer released from full lift")
    led.add("cam_dwell_deg", dwell, "deg", "delta >= margin x 360 n_cam t_f", f"margin {cr['hammer_fall_time_margin']}, rounded up to 5 deg")
    led.add("cam_rise_deg", beta, "deg", "beta = 360/n_lobes - delta", "cycloidal rise, then a cliff that drops the hammer")
    r_roll = float(hd["cam_roller_diameter_m"]) / 2
    w_roll = float(hd["cam_roller_width_m"])
    r_m = float(hd["hammer_od_m"]) / 2 + 0.003 + w_roll / 2   # rollers clear the hammer by a 3 mm stationary spigot gap
    eta_cam = float(perc["cam_efficiency"])
    br = math.radians(beta)
    prof, torques = [], []
    n_s = 360
    for i in range(n_s + 1):
        th = 360.0 / lobes * i / n_s
        if th <= beta:
            u = th / beta
            zc = h_c * (u - math.sin(2 * math.pi * u) / (2 * math.pi))
            dz = h_c / br * (1 - math.cos(2 * math.pi * u))
        else:
            zc, dz = 0.0, 0.0
        zh = max(0.0, zc - gap)
        torques.append(k * zh * dz / eta_cam)
        prof.append((round(th, 2), round(zc * 1e3, 4)))
    t_peak = max(torques)
    t_rms = math.sqrt(sum(t * t for t in torques[:-1]) / n_s)
    t_mean = sum(torques[:-1]) / n_s
    phi_max = math.degrees(math.atan(2 * h_c / br / r_m))
    led.add("cam_mean_radius_mm", round(r_m * 1e3, 1), "mm", "R_m = r_hammer + spigot gap + w_roller/2", "roller track radius")
    led.add("cam_pressure_angle_deg", round(phi_max, 1), "deg", "phi = atan((ds/dtheta)_max / R_m), (ds/dtheta)_max = 2 h_c / beta",
            f"<= {cr['cam_max_pressure_angle_deg']} deg (cycloidal: zero acceleration at both ends of the rise)")
    led.add("cam_torque_peak_nm", round(t_peak, 3), "N m", "T = k s(theta) s'(theta) / eta_cam", f"RMS {t_rms:.3f}, mean {t_mean:.3f} N m")
    led.add("cam_torque_mean_check_nm", round(lobes * 0.5 * k * h * h / (2 * math.pi * eta_cam), 3), "N m",
            "T_mean = n_lobes E / (2 pi eta_cam)", "energy balance agrees with the integrated profile")
    gs = p["materials"]["gear_steel"]
    e_star = float(gs["youngs_modulus_pa"]) / (2 * (1 - float(gs["poisson"]) ** 2))
    f_roll = k * h / 2
    p_roll = math.sqrt(f_roll * e_star / (math.pi * r_roll * w_roll))
    led.add("cam_roller_contact_mpa", round(p_roll / 1e6, 0), "MPa", "p = sqrt(F E* / (pi R L))",
            f"F = k h / 2 = {f_roll:.0f} N per roller (two rollers); allowable {float(gs['contact_strength_pa'])/1e6/cr['gear_contact_fos']:.0f} MPa")

    # Percussive drive: secondary BLDC -> 18T pinion -> 54T ring gear on the cam
    z_pp, z_cg = int(hd["percussive_pinion_teeth"]), int(hd["cam_gear_teeth"])
    i_c = z_cg / z_pp
    cam_rpm_abs = (cam_rps + float(rot["nominal_rpm"]) / 60) * 60
    b_cam = hd["cam_bearing"]
    rim = 0.004
    t_pp_peak = t_peak / i_c / eta_sp

    def cg_check(m):
        return [_mesh(p, z_pp, z_cg, m, t_pp_peak / (m * z_pp / 2), math.pi * m * z_pp * cam_rpm_abs * i_c / 60)]

    m_min_cg = (float(b_cam["D_m"]) + 2 * rim) / (z_cg - 2.5)   # root circle clears the cam bearing housing
    cg = _pick_module(p, cg_check, m_min_cg)
    m_cg = cg["m"]
    led.add("cam_gear_module_mm", m_cg * 1e3, "mm", "m >= (D_brg + 2 rim) / (Z - 2.5) and AGMA checks",
            f"{z_pp}T/{z_cg}T ring gear on the cam; packaging-driven, bending FOS {cg['meshes'][0]['fos_bending']}")
    t_pm = t_rms / i_c / eta_sp
    d_rot2 = (4 * t_pm / (math.pi * trv * lam)) ** (1 / 3)
    led.add("percussive_motor_torque_nm", round(t_pm, 3), "N m", "T_m = T_cam,rms / (i eta)",
            f"peak {t_peak / i_c / eta_sp:.3f} N m; {cam_rpm_abs * i_c:.0f} rpm at nominal spindle speed")
    led.add("percussive_motor_stator_od_mm", round(d_rot2 / float(hd["motor_split_ratio"]) * 1e3, 1), "mm",
            "D_s = (4 T / (pi TRV lambda))^(1/3) / split", "sized on RMS (thermal) torque")
    out["cam"] = {"rise_deg": beta, "dwell_deg": dwell, "rise_m": h_c, "gap_m": gap, "mean_radius_m": r_m,
                  "profile_deg_mm": prof, "torque_peak_nm": t_peak, "torque_rms_nm": t_rms,
                  "pressure_angle_deg": phi_max, "roller_contact_pa": p_roll, "roller_r_m": r_roll, "roller_w_m": w_roll,
                  "lobes": lobes}
    out["percussive"] = {"pinion_teeth": z_pp, "cam_gear_teeth": z_cg, "module_m": m_cg, "motor_torque_nm": t_pm,
                         "rotor_d_m": d_rot2, "stator_od_m": d_rot2 / float(hd["motor_split_ratio"]),
                         "rotor_l_m": lam * d_rot2, "cam_rpm": cam_rpm_abs, "mesh": cg["meshes"][0]}

    # ---- Spline and spindle -----------------------------------------------------------------------
    d_maj, m_s, z_sp = float(hd["spline_major_m"]), float(hd["spline_module_m"]), int(hd["spline_teeth"])
    d_pitch = m_s * z_sp
    l_e = float(hd["anvil_spline_engagement_m"])
    p_spl = 2 * t_out / (d_pitch * z_sp * 0.9 * m_s * l_e * float(hd["spline_load_sharing"]))
    led.add("spline_flank_pressure_mpa", round(p_spl / 1e6, 2), "MPa", "p = 2 T / (d z h_w L K), h_w = 0.9 m",
            f"{hd['spline']}, {l_e*1e3:.0f} mm anvil engagement; allowable {float(hd['spline_allowable_pressure_pa'])/1e6:.0f} MPa (sliding hub)")
    d_root = d_maj - 2.2 * m_s
    d_i = float(hd["spindle_bore_m"])
    tau_sp = 16 * t_out * d_root / (math.pi * (d_root ** 4 - d_i ** 4))
    sy = float(p["materials"]["auger_steel"]["yield_strength_pa"])
    led.add("spindle_torsion_fos", round(sy / math.sqrt(3) / tau_sp, 1), "-", "tau = 16 T d / (pi (d^4 - d_i^4)); n = S_y / (sqrt3 tau)",
            f"at the spline root d = {d_root*1e3:.2f} mm, bore {d_i*1e3:.0f} mm, tau = {tau_sp/1e6:.1f} MPa")
    out["spline"] = {"pitch_d_m": d_pitch, "root_d_m": d_root, "flank_pressure_pa": p_spl}

    # ---- Axial stack of the head (mm above the carriage underside) -------------------------------
    sb, ub = hd["spindle_bearing"], hd["upper_bearing"]
    st: Dict[str, float] = {}
    st["base_plate"] = 10.0
    st["housing_bottom"] = 12.0
    st["nose_top"] = 22.0
    st["anvil_bottom"] = -12.0                               # anvil nose and dust cap sit below the underside
    st["hex_top"] = 40.0                                      # auger hex shank seats up to here (interface)
    st["anvil_shoulder"] = st["nose_top"] + float(hd["anvil_float_m"]) * 1e3
    st["web_top"] = st["hex_top"] + 6.0
    st["anvil_top"] = st["web_top"] + l_e * 1e3
    st["hammer_top"] = st["anvil_top"] + float(hd["hammer_length_m"]) * 1e3
    st["roller_z"] = 0.5 * (st["anvil_top"] + st["hammer_top"])
    st["track_base"] = st["roller_z"] - r_roll * 1e3 - gap * 1e3
    b_c = float(hd["face_width_to_module"]) * m_cg * 1e3
    st["cam_gear_bottom"] = st["track_base"] - b_c
    spring_free = hammer["spring"]["free_length_mm"]
    st["spring_top"] = st["hammer_top"] + spring_free
    st["seat_top"] = st["spring_top"] + 6.0
    st["pair_bottom"] = st["seat_top"] + 2.0
    st["pair_top"] = st["pair_bottom"] + 2 * float(sb["B_m"]) * 1e3
    st["slip_bottom"] = st["pair_top"] + 3.0
    st["slip_top"] = st["slip_bottom"] + 26.0
    st["upper_bottom"] = st["slip_top"] + 2.0
    st["upper_top"] = st["upper_bottom"] + float(ub["B_m"]) * 1e3
    b_sp = float(hd["face_width_to_module"]) * m_sp * 1e3
    st["gear_bottom"] = st["upper_top"] + 3.0
    st["gear_top"] = st["gear_bottom"] + b_sp
    b_pl = float(hd["face_width_to_module"]) * m_pl * 1e3
    st["planetary_top"] = st["gear_top"] + 2.0 + 2 * b_pl + 10.0
    st["motor_top"] = st["planetary_top"] + lam * d_rot * 1e3 + 24.0     # stack + end bells / encoder
    st["housing_top"] = st["motor_top"] + float(hd["housing_wall_m"]) * 1e3 + 2.0
    st["load_cell_top"] = st["housing_top"] + 20.0
    st["yoke_top"] = st["load_cell_top"] + 12.0
    out["stack_mm"] = {k2: round(v2, 2) for k2, v2 in st.items()}
    # Plan-view packaging: housing radius over the cam ring gear; the Tr16 leadscrew must clear it
    wall = float(hd["housing_wall_m"]) * 1e3
    r_cam_tip = m_cg * (z_cg + 2) / 2 * 1e3
    r_low = r_cam_tip + 1.5 + wall
    out["plan_mm"] = {"housing_r_lower": round(r_low, 2), "cam_tip_r": round(r_cam_tip, 2),
                      "leadscrew_y": math.ceil((r_low + 8.0 + 2.0) / 5.0) * 5.0,
                      "rotary_axis_x": round(a_sp * 1e3, 2),
                      "percussive_axis_y": -round(m_cg * (z_pp + z_cg) / 2 * 1e3, 2)}
    led.add("leadscrew_offset_mm", out["plan_mm"]["leadscrew_y"], "mm", "y_ls >= R_housing + r_screw + 2 mm",
            "Tr16 screw passes the cam-gear housing; bronze nut clamped in the top yoke")
    led.add("drill_head_height_mm", round(st["yoke_top"], 1), "mm", "sum of the derived axial stack",
            "carriage underside to the top of the load-cell yoke")

    # ---- Bearings ---------------------------------------------------------------------------------
    life_req = float(hd["design_operating_hours"]) * float(cr["mechanism_life_factor"])
    n_nom = float(rot["nominal_rpm"])
    d_g = m_sp * z_g
    ft_g = t_out / (d_g / 2)
    fg = ft_g / math.cos(math.radians(float(hd["pressure_angle_deg"])))
    z_pair = 0.5 * (st["pair_bottom"] + st["pair_top"])
    z_up = 0.5 * (st["upper_bottom"] + st["upper_top"])
    z_gear = 0.5 * (st["gear_bottom"] + st["gear_top"])
    r_up = fg * (z_gear - z_pair) / (z_up - z_pair)          # overhung gear above the upper bearing
    r_pair = fg * (z_gear - z_up) / (z_up - z_pair)
    wob_max = max(v["wob_limit_n"] for v in wob.values())
    fa = wob_max + k * h
    c_pair, c0_pair = 1.62 * float(sb["C_n"]), 2.0 * float(sb["C0_n"])
    p_pair = r_pair + 0.92 * fa if fa / max(r_pair, 1e-9) <= 0.68 else 0.67 * r_pair + 1.41 * fa
    p0_pair = r_pair + 0.76 * fa
    l_pair = _l10h(c_pair, p_pair, n_nom)
    led.add("spindle_bearing_l10_h", round(l_pair, 0), "h", "L10h = (C/P)^3 10^6 / (60 n);  P = 0.67 F_r + 1.41 F_a",
            f"{sb['name']}: C = 1.62 C_single; F_a = WOB_lim + k h = {fa:.0f} N, F_r = {r_pair:.0f} N; required {life_req:.0f} h")
    led.add("spindle_bearing_s0", round(c0_pair / p0_pair, 1), "-", "s0 = C0 / (F_r + 0.76 F_a)", f">= {cr['bearing_static_fos']}")
    l_up = _l10h(float(ub["C_n"]), r_up, n_nom)
    led.add("upper_bearing_l10_h", round(l_up, 0), "h", "L10h = (C/F_r)^3 10^6 / (60 n)",
            f"{ub['name']}, floating; F_r = {r_up:.0f} N from the overhung {z_g}T gear (F = {fg:.0f} N)")
    # Cam bearing: spring reaction (axial) + cam-gear mesh (radial) at peak cam torque
    r_cg = m_cg * z_cg / 2
    fr_c = t_peak / r_cg / math.cos(math.radians(float(hd["pressure_angle_deg"])))
    fa_c = k * h
    ratio_f0 = float(b_cam["f0"]) * fa_c / float(b_cam["C0_n"])
    e_c, y_c = _interp(DGBB_AXIAL, ratio_f0, 1), _interp(DGBB_AXIAL, ratio_f0, 2)
    p_c = 0.56 * fr_c + y_c * fa_c if fa_c / fr_c > e_c else fr_c
    l_cam = _l10h(float(b_cam["C_n"]), p_c, cam_rpm_abs)
    p0_c = max(fr_c, 0.6 * fr_c + 0.5 * fa_c)
    led.add("cam_bearing_l10_h", round(l_cam, 0), "h", "P = 0.56 F_r + Y F_a  (Y from f0 F_a / C0)",
            f"{b_cam['name']}: F_a = k h = {fa_c:.0f} N, F_r = {fr_c:.0f} N, Y = {y_c:.2f}, {cam_rpm_abs:.0f} rpm; s0 = {float(b_cam['C0_n'])/p0_c:.1f}")
    out["bearings"] = {"required_h": life_req, "spindle_l10_h": l_pair, "spindle_s0": c0_pair / p0_pair,
                       "upper_l10_h": l_up, "cam_l10_h": l_cam, "cam_s0": float(b_cam["C0_n"]) / p0_c,
                       "spindle_fa_n": fa, "spindle_fr_n": r_pair, "upper_fr_n": r_up}

    # ---- WOB load cell and leadscrew nut ------------------------------------------------------------
    need = float(cr["load_cell_overload_factor"]) * wob_max
    cap = min(c for c in hd["load_cell_capacities_n"] if c >= need)
    err = float(hd["load_cell_error_fs"]) * cap
    led.add("load_cell_capacity_n", cap, "N", "min standard range >= factor x WOB_lim,max",
            f"{need:.0f} N needed; error {err:.2f} N vs {zf['contact_detect_wob_n']} N contact threshold")
    g_max = max(float(e["gravity_m_s2"]) for e in p["environments"].values())
    f_nut = wob_max + float(zf["moving_mass_kg"]) * g_max
    d_ls, lead = 16.0e-3, float(zf["lead_mm"]) / 1e3
    d2 = float(zf["screw_mean_diameter_mm"]) / 1e3
    l_nut = float(hd["leadscrew_nut_length_to_d"]) * d_ls
    p_nut = f_nut / (math.pi * d2 * 0.5 * lead * (l_nut / lead))
    mu = float(zf["thread_friction_coeff"])
    lam_l = math.atan(lead / (math.pi * d2))
    self_lock = mu / math.cos(math.radians(15.0)) > math.tan(lam_l)
    led.add("leadscrew_nut_pressure_mpa", round(p_nut / 1e6, 2), "MPa", "p = F / (pi d_2 h_1 n_t)",
            f"Tr16x4 bronze nut {l_nut*1e3:.0f} mm long, F = {f_nut:.0f} N; allowable {float(hd['nut_allowable_pressure_pa'])/1e6:.0f} MPa")
    led.add("leadscrew_self_locking", self_lock, "-", "mu / cos(alpha/2) > tan(lambda)",
            f"lead angle {math.degrees(lam_l):.2f} deg: WOB cannot back-drive the screw when the feed motor is unpowered")
    out["load_cell_n"] = cap
    out["load_cell_error_n"] = err
    out["nut"] = {"pressure_pa": p_nut, "length_m": l_nut, "self_locking": self_lock}
    return out
