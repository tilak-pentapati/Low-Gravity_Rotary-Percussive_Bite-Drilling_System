"""
Deployable drill-mast derivations: mast spine section and material, latch-post height, rail-support brackets,
stowed (launch) pose, hold-downs, hinge, deployed-position latch, deployment actuator, leadscrew checks, lander
deck length, leg footprint and the tilt (lift-off) check of a side-mounted drill. Called from src/design.py;
every result is written to the derivation ledger.

Architecture (x right, y aft toward the lander, z up; drill axis at x = y = 0):
  spine      closed rectangular tube aft of the Tr16x4 leadscrew. Discrete rail-support brackets bolted to its
             fore face hold the two Ø16 rails; their necks pass the slot of open LM16UU-OP bushings in the carriage
  hinge      axis along x at the spine aft face on the deck fore edge (z = deck top); two Ti lugs on the spine
             sides in deck clevises; strain-wave rotary actuator on the hinge axis
  latch      two posts on the deck beside the spine; spring latch pins engage spine lugs h_l above the hinge
  stowed     mast rotated 90 deg about the hinge so it lies aft over the deck, carriage parked at the derived
             stow pose (bit through the guide bushing), SMA-released frangible-bolt hold-down(s) on the deck
  sequence   release hold-downs -> carriage to home -> deploy 90 deg -> latch -> release the brush arm

Beam models (Euler-Bernoulli finite elements along the spine, z from the foot plate to the crosshead):
  deployed  fore-aft plane: hinge pin + latch pin;   side plane: hinge clamp (lugs spread in x) + latch pin
  stowed    deck-normal plane: hinge pin + hold-down pin(s);  lateral plane: hinge clamp + hold-down pin(s)
  rails, brackets and leadscrew add mass, not stiffness (conservative); carriage, crosshead and foot are lumped.
  The leadscrew, the auger string and each rail span are checked as their own beams at launch.
Alignment: the string is clamped in the carriage, so the offset at the guide bushing is
  e = w(carriage) + phi(carriage) L_cb - w(bushing),  phi = spine slope + rail-flex tilt of the bushing pairs.
Launch loads: max(quasi-static load factor, Miles 3-sigma response g = 3 sqrt(pi/2 f Q PSD(f))) on each axis.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.linalg as sla

G0 = 9.80665


# ------------------------------------------------------------------------------
# Finite-element beam: unit matrices so a section / mass change is a rescale
# ------------------------------------------------------------------------------
class Beam:
    """Hermite beam elements on [s0, s1]; props(s) -> (EI, m per length, c = outer fibre distance)."""

    def __init__(self, s0: float, s1: float, n_el: int, props: Callable[[float], Tuple[float, float, float]]) -> None:
        self.s = np.linspace(s0, s1, n_el + 1)
        n = 2 * len(self.s)
        self.K, self.M = np.zeros((n, n)), np.zeros((n, n))
        self.el: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        for e in range(n_el):
            L = self.s[e + 1] - self.s[e]
            ei, m, c = props(0.5 * (self.s[e] + self.s[e + 1]))
            k = ei / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                        [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
            mm = m * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                         [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
            dof = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
            self.K[np.ix_(dof, dof)] += k
            self.M[np.ix_(dof, dof)] += mm
            self.el.append((dof, k, mm, c, ei))
        self.rigid = np.zeros(n)
        self.rigid[0::2] = 1.0

    def node(self, s: float) -> int:
        return int(np.argmin(np.abs(self.s - s)))

    def free(self, supports: Sequence[Tuple[float, str]]) -> Tuple[List[int], List[int]]:
        fixed = set()
        for s, kind in supports:
            fixed.add(2 * self.node(s))
            if kind == "clamp":
                fixed.add(2 * self.node(s) + 1)
        return sorted(fixed), [i for i in range(self.K.shape[0]) if i not in fixed]

    def with_lumps(self, M: np.ndarray, lumps: Sequence[Tuple[float, float]]) -> np.ndarray:
        M = M.copy()
        for s, m in lumps:
            M[2 * self.node(s), 2 * self.node(s)] += m
        return M

    def f1(self, supports, lumps=(), k_scale: float = 1.0, M: np.ndarray = None) -> float:
        _, f = self.free(supports)
        Mt = self.with_lumps(self.M if M is None else M, lumps)
        ix = np.ix_(f, f)
        w2 = sla.eigh(k_scale * self.K[ix], Mt[ix], eigvals_only=True, subset_by_index=[0, 0])[0]
        return math.sqrt(max(w2, 0.0)) / (2 * math.pi)

    def static(self, supports, lumps=(), accel: float = 0.0, loads: Sequence[Tuple[float, float, float]] = ()):
        """Inertial load accel x (distributed + lumped mass) plus point loads (s, F, M): deflection, slope, moments, reactions."""
        fixed, f = self.free(supports)
        M = self.with_lumps(self.M, lumps)
        F = accel * (M @ self.rigid)
        for s, fz, mo in loads:
            F[2 * self.node(s)] += fz
            F[2 * self.node(s) + 1] += mo
        u = np.zeros_like(F)
        u[f] = np.linalg.solve(self.K[np.ix_(f, f)], F[f])
        react = self.K @ u - F
        sig_k = 0.0
        for dof, k, mm, c, ei in self.el:
            fe = k @ u[dof] - accel * (mm @ self.rigid[dof])
            sig_k = max(sig_k, max(abs(fe[1]), abs(fe[3])) * c / ei)     # sigma / E
        return {"w": u[0::2], "phi": u[1::2], "react": {int(i): float(react[i]) for i in fixed}, "sig_over_e": sig_k}


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
def box_section(b: float, h: float, t: float) -> Dict[str, float]:
    """Rectangular tube b (x) x h (y) x t: area, I about x (fore-aft bending), I about y (side bending)."""
    bi, hi = b - 2 * t, h - 2 * t
    return {"A": b * h - bi * hi, "Ix": (b * h ** 3 - bi * hi ** 3) / 12, "Iy": (h * b ** 3 - hi * bi ** 3) / 12}


def psd_at(p: Dict[str, Any], f: float) -> float:
    """GEVS random PSD (g^2/Hz), log-log interpolation between break points; flat beyond the ends."""
    pts = p["criteria"]["random_vibration_psd"]
    if f <= pts[0][0]:
        return float(pts[0][1])
    for (f0, a0), (f1, a1) in zip(pts[:-1], pts[1:]):
        if f <= f1:
            return float(a0 * (f / f0) ** (math.log(a1 / a0) / math.log(f1 / f0)))
    return float(pts[-1][1])


def miles_3sigma_g(p: Dict[str, Any], f: float) -> float:
    return 3.0 * math.sqrt(math.pi / 2 * f * float(p["criteria"]["random_vibration_q"]) * psd_at(p, f))


def in_band(f: float, forcing: Sequence[Tuple[float, float]], sep: float) -> bool:
    return any(lo / (1 + sep) <= f <= hi * (1 + sep) for lo, hi in forcing)


def max_sun_elevation_deg(p: Dict[str, Any]) -> float:
    """Highest sun elevation either wing must track: 90 - |lat - dec| (dec as in src/solar.py)."""
    out = 0.0
    for env_id, env in p["environments"].items():
        if env_id == "mars":
            dec = math.degrees(math.asin(math.sin(math.radians(env["obliquity_deg"])) * math.sin(math.radians(env["season_ls_deg"]))))
        else:
            dec = -float(env["sun_declination_deg"]) * (1 if env["latitude_deg"] < 0 else -1)
        out = max(out, 90.0 - abs(float(env["latitude_deg"]) - dec))
    return out


def pick_pin(p: Dict[str, Any], f_lug: float, t_lug: float) -> Dict[str, float]:
    """Smallest clevis pin: double shear (ultimate) and Ti lug bearing (yield) at the per-lug load."""
    cr, pin_m, lug_m = p["criteria"], p["materials"]["auger_steel"], p["materials"]["mast_titanium"]
    tau_u = float(pin_m["ultimate_strength_pa"]) / math.sqrt(3)
    for d in p["drill_mast"]["hinge_pin_options_m"]:
        tau = f_lug / (2 * math.pi / 4 * d ** 2)
        br = f_lug / (d * t_lug)
        if tau * float(cr["structural_fos_ultimate"]) <= tau_u and br * float(cr["structural_fos_yield"]) <= float(lug_m["yield_strength_pa"]):
            return {"d_m": float(d), "tau_mpa": tau / 1e6, "bearing_mpa": br / 1e6,
                    "fos_shear": tau_u / tau, "fos_bearing": float(lug_m["yield_strength_pa"]) / br}
    raise AssertionError("no hinge/latch pin option passes")


def leg_reactions(m_g: float, wob: float, y_g: float, y_near: float, y_far: float) -> Tuple[float, float]:
    """Per-leg normal loads (near, far) of a 4-leg lander symmetric in x, upward WOB reaction at y = 0."""
    span = y_far - y_near
    n_near = (m_g * (y_far - y_g) - wob * (y_far - 0.0)) / span
    return n_near / 2, (m_g - wob) / 2 - n_near / 2


# ------------------------------------------------------------------------------
# Derivation
# ------------------------------------------------------------------------------
def derive_mast(p: Dict[str, Any], led, wob: Dict[str, Any], hammer: Dict[str, Any], head: Dict[str, Any],
                string: Dict[str, Any], brush: Dict[str, Any]) -> Dict[str, Any]:
    mc, li, cr, lay = p["drill_mast"], p["lander_interface"], p["criteria"], p["cad_layout"]
    mats = p["materials"]
    al, fs = mats["mast_aluminium"], mats["feed_steel"]
    zf, rot, hd = p["actuators"]["z_axis_feed"], p["actuators"]["rotary"], p["drill_head"]
    sep = float(p["auger_string"]["resonance_separation"])
    g_env = {k: float(e["gravity_m_s2"]) for k, e in p["environments"].items()}
    g_max = max(g_env.values())
    fos_y, fos_u = float(cr["structural_fos_yield"]), float(cr["structural_fos_ultimate"])
    mga = 1.0 + float(mc["mass_growth_allowance"])
    qs = float(cr["launch_quasi_static_g"])
    f_req = float(cr["launch_min_frequency_hz"])
    out: Dict[str, Any] = {}

    # ---- Fixed geometry (m) -------------------------------------------------------------------------
    rail_x = float(mc["rail_x_m"])
    r_rail = float(hd["linear_bearing"]["d_m"]) / 2
    r_lb, l_lb = float(hd["linear_bearing"]["D_m"]) / 2, float(hd["linear_bearing"]["L_m"])
    stack = head["stack_mm"]
    y_ls = head["plan_mm"]["leadscrew_y"] / 1e3
    r_flange = 0.018                                         # Ø36 Tr16 nut flange (src/cad_head.py)
    c_run = float(mc["running_clearance_m"])
    t_neck, t_br, d_br = float(mc["rail_bracket_neck_m"]), float(mc["rail_bracket_thickness_m"]), float(mc["rail_bracket_depth_m"])
    y_br0 = y_ls + r_flange + c_run                          # bracket body fore face (aft of the carriage and nut flange)
    y_sf = y_br0 + d_br                                      # spine fore face
    z_deck = float(li["deck_top_height_m"])
    z_h = z_deck + float(mc["stow_gap_m"])                   # hinge axis: the stowed spine aft face clears the deck by the gap
    z_ft = float(lay["foot_top_m"])
    t_ft, t_xh = float(mc["foot_plate_thickness_m"]), float(mc["crosshead_thickness_m"])
    z_f0 = z_ft - t_ft
    home_tip, l_aug = float(lay["home_tip_height_m"]), float(lay["auger_length_m"])
    zc_home = home_tip + l_aug
    env_h = float(mc["carriage_envelope_m"])
    assert stack["yoke_top"] / 1e3 <= env_h, "drill head taller than the carriage envelope"
    z_xh0 = zc_home + env_h + c_run
    z_top = z_xh0 + t_xh
    l_mot, d_mot = float(mc["feed_motor_length_m"]), float(mc["feed_motor_od_m"])
    l_rail = z_xh0 - z_ft
    half_w = rail_x + r_lb + 0.004                           # crosshead / foot half-width = rail block outer face
    # Foot plate: rectangle from the hinge to the carousel axis plus a disc over the sample carousel (the plate is the
    # cups' sealing cover); the carousel hangs under it
    car = brush["carousel"]
    y_foot_fore = -car["axis_offset_m"]
    r_disc = car["plate_disc_r_m"]
    xs, ys = np.meshgrid(np.linspace(-r_disc, r_disc, 401), np.linspace(-r_disc, r_disc, 401))
    in_disc = xs ** 2 + ys ** 2 <= r_disc ** 2
    extra = in_disc & ~((np.abs(xs) <= half_w) & (ys >= 0.0))     # disc area not already inside the rectangle
    cell = (2 * r_disc / 400) ** 2
    a_lobe = float(extra.sum()) * cell
    y_lobe = y_foot_fore + float(ys[extra].mean())
    m_car = car["mass_kg"]
    target = float(p["system"]["target_depth_m"])
    m_str = float(string["beam"]["mass_kg"])
    m_mov = float(zf["moving_mass_kg"])
    m_head = m_mov - m_str
    l_bp = (stack["load_cell_top"] - stack["base_plate"]) / 1e3 - l_lb      # centre distance of the bushing pairs
    led.add("mast_spine_fore_face_mm", round(y_sf * 1e3, 1), "mm", "y = y_ls + r_flange + c + d_bracket",
            f"rail brackets clear the Tr16 nut flange by {c_run*1e3:.0f} mm; carriage bushing pairs {l_bp*1e3:.0f} mm apart")

    # ---- Line masses (kg/m) independent of the spine ------------------------------------------------
    d2 = float(zf["screw_mean_diameter_mm"]) / 1e3
    lead = float(zf["lead_mm"]) / 1e3
    d3 = 0.016 - 2 * (0.5 * lead + 0.00025)                  # ISO 2904 Tr16x4 minor diameter, a_c = 0.25 mm
    e_fs, rho_fs = float(fs["youngs_modulus_pa"]), float(fs["density_kg_m3"])
    mu_rails = 2 * rho_fs * math.pi * r_rail ** 2
    mu_screw = rho_fs * math.pi / 4 * d2 ** 2
    m_bracket = float(al["density_kg_m3"]) * t_br * (2 * t_neck * (y_br0 - r_rail) + (2 * rail_x + t_neck) * d_br) * mga
    rho_act = float(mc["actuator_mean_density_kg_m3"])
    m_motor = rho_act * math.pi / 4 * d_mot ** 2 * l_mot * mga

    def plate_masses(h_sec: float) -> Dict[str, float]:
        y_h_ = y_sf + h_sec
        m_x = float(al["density_kg_m3"]) * t_xh * 2 * half_w * (y_h_ + r_lb + 0.004)
        m_rect = float(al["density_kg_m3"]) * t_ft * 2 * half_w * (y_h_ - y_foot_fore)
        m_lobe = float(al["density_kg_m3"]) * t_ft * a_lobe
        y_foot = (m_rect * 0.5 * (y_h_ + y_foot_fore) + m_lobe * y_lobe + m_car / mga * y_foot_fore) / (m_rect + m_lobe + m_car / mga)
        return {"crosshead": m_x * mga + m_motor, "foot": (m_rect + m_lobe) * mga + m_car, "y_foot": y_foot}

    # ---- Rail-support brackets: each rail span must clear the launch minimum on its own ----------------
    ei_r = e_fs * math.pi / 4 * r_rail ** 4
    l_span_max = math.sqrt(math.pi / (2 * f_req) * math.sqrt(ei_r / (rho_fs * math.pi * r_rail ** 2)))
    n_br_min = max(0, math.ceil(l_rail / l_span_max) - 1)
    f_end = math.pi / (2 * l_rail ** 2) * math.sqrt(ei_r / (rho_fs * math.pi * r_rail ** 2))
    led.add("rail_span_max_mm", round(l_span_max * 1e3, 0), "mm", "f = (pi / 2 L^2) sqrt(EI / m) >= f_launch",
            f"Ø16 rail between supports; end-supported over {l_rail*1e3:.0f} mm it would be {f_end:.0f} Hz")

    # ---- Stow pose: carriage lowered so the bit passes the guide bushing ----------------------------
    bu = string["bushing"]
    z_b0, z_b1 = bu["z0_m"], bu["z0_m"] + bu["length_m"]
    z_bm = 0.5 * (z_b0 + z_b1)
    nut_off = stack["yoke_top"] / 1e3 - head["nut"]["length_m"] / 2 + 0.006   # nut centre above the carriage underside
    i3 = math.pi / 64 * d3 ** 4
    screw = Beam(z_ft, z_xh0, 60, lambda s: (e_fs * i3, mu_screw, d3 / 2))
    segs = string["segments"]

    def seg_props(s: float) -> Tuple[float, float, float]:
        for s0, s1, pr in segs:
            if s0 <= s <= s1:
                return pr["EI"], pr["m"], pr["c"]
        return segs[-1][2]["EI"], segs[-1][2]["m"], segs[-1][2]["c"]

    clamp = float(string["stations_m"]["clamp"])
    strb = Beam(0.0, clamp, 90, seg_props)
    crown_top = float(string["stations_m"]["crown_top"])
    step = float(mc["stow_pose_step_m"])
    best = None
    f_home_screw = screw.f1([(z_ft, "pin"), (zc_home + nut_off, "pin"), (z_xh0, "clamp")])
    f_home_str = strb.f1([(clamp, "clamp")])
    for delta in np.arange(home_tip - z_b0 + crown_top, home_tip + target + 1e-9, step):
        tip = home_tip - delta
        fs1 = screw.f1([(z_ft, "pin"), (zc_home - delta + nut_off, "pin"), (z_xh0, "clamp")])
        fs2 = strb.f1([(z_b0 - tip, "pin"), (z_b1 - tip, "pin"), (clamp, "clamp")])
        if best is None or min(fs1, fs2) > best[0]:
            best = (min(fs1, fs2), float(delta), fs1, fs2)
    _, d_stow, f_screw_st, f_str_st = best
    tip_st = home_tip - d_stow
    zc_stow = zc_home - d_stow
    led.add("mast_stow_carriage_drop_mm", round(d_stow * 1e3, 0), "mm", "argmax_delta min(f_screw, f_string)",
            f"launch pose: carriage {d_stow*1e3:.0f} mm below home, bit {(z_f0 - tip_st)*1e3:.0f} mm through the foot. The nut splits "
            f"the leadscrew and the guide bushing props the string (at home: screw {f_home_screw:.0f} Hz, string {f_home_str:.0f} Hz)")

    # Secondary members at launch: quasi-static or Miles 3-sigma, whichever is larger
    sy_fs, st_m = float(fs["yield_strength_pa"]), p["materials"]["auger_steel"]
    g_scr = max(qs, miles_3sigma_g(p, f_screw_st))
    sc = screw.static([(z_ft, "pin"), (zc_stow + nut_off, "pin"), (z_xh0, "clamp")], accel=g_scr * G0)
    gap_screw = y_br0 - y_ls - 0.008
    sig_scr = sc["sig_over_e"] * e_fs
    sag_scr = float(np.max(np.abs(sc["w"])))
    led.add("leadscrew_stowed_f1_hz", round(f_screw_st, 1), "Hz", "K phi = w^2 M phi: fixed top bearing, nut, support bearing",
            f"local mode (not the stowed primary); launch load {g_scr:.1f} g (Miles 3-sigma vs {qs:.0f} g quasi-static)")
    led.add("leadscrew_launch_stress_mpa", round(sig_scr / 1e6, 1), "MPa", "sigma = M c / I (minor diameter d3)",
            f"yield FOS {sy_fs/sig_scr:.1f} (AISI 1045); peak sag {sag_scr*1e3:.2f} mm vs {gap_screw*1e3:.0f} mm to the rail brackets")
    g_strg = max(qs, miles_3sigma_g(p, f_str_st))
    ss = strb.static([(z_b0 - tip_st, "pin"), (z_b1 - tip_st, "pin"), (clamp, "clamp")], accel=g_strg * G0)
    sig_str = ss["sig_over_e"] * float(st_m["youngs_modulus_pa"])
    led.add("string_stowed_f1_hz", round(f_str_st, 1), "Hz", "auger-string FE model with the bushing as launch support",
            f"local mode; launch load {g_strg:.1f} g, peak bending {sig_str/1e6:.0f} MPa (yield FOS "
            f"{float(st_m['yield_strength_pa'])/sig_str:.1f}), sag {np.max(np.abs(ss['w']))*1e3:.2f} mm")
    out["stow"] = {"carriage_drop_m": d_stow, "tip_z_m": tip_st, "screw_f1_hz": f_screw_st, "string_f1_hz": f_str_st,
                   "screw_home_f1_hz": f_home_screw, "string_home_f1_hz": f_home_str,
                   "screw_launch_g": g_scr, "screw_stress_mpa": sig_scr / 1e6, "screw_sag_mm": sag_scr * 1e3,
                   "screw_gap_mm": gap_screw * 1e3, "string_launch_g": g_strg, "string_stress_mpa": sig_str / 1e6}

    # ---- Loads on the deployed mast ---------------------------------------------------------------
    wob_lim = max(w["wob_limit_n"] for w in wob.values())
    r_g = float(p["auger_geometry"]["gauge_diameter_m"]) / 2
    f_side = float(rot["max_safe_torque_nm"]) / r_g          # one cutter carrying the full torque: unbalanced side force
    l_cb0 = zc_home - home_tip - z_bm                        # carriage clamp to bushing at the start of the hole
    r_carr = 1.5 * z_bm / l_cb0 * f_side                     # propped cantilever with a tip load (Roark)
    r_bush = f_side + r_carr
    led.add("bit_side_force_n", round(f_side, 0), "N", "F = T_max / r_gauge",
            "one insert hung up on a clast at the torque limit: the largest unbalanced lateral force at the bit")
    led.add("bit_side_reaction_carriage_n", round(r_carr, 1), "N", "R_A = 3 a F / (2 L), R_B = F + R_A",
            f"string clamped at the carriage, propped by the bushing {z_bm*1e3:.0f} mm above the bit: the foot takes {r_bush:.0f} N")
    forcing = [(float(rot["min_rpm"]) / 60, float(rot["max_rpm"]) / 60),
               (float(p["actuators"]["percussive"]["cam_speed_rps"]),) * 2,
               (hammer["blow_frequency_hz"],) * 2]
    drill_poses = [zc_home - home_tip - d for d in (0.0, 0.5 * target, target)]
    pull = float(string["shank"]["pull_n"])
    clr = float(p["auger_string"]["guide_bushing_radial_clearance_m"])
    # Rail-flex tilt of the carriage: couple WOB y_ls (+ side reaction) on the bushing pairs, worst at mid-span
    def rail_tilt(n_br: int) -> float:
        span = l_rail / (n_br + 1)
        f_pair = (wob_lim * y_ls + r_carr * l_bp / 2) / l_bp / 2           # per rail, per bushing pair
        return 2 * f_pair * span ** 3 / (48 * ei_r) / l_bp

    # ---- Section / material / latch-height / bracket search ---------------------------------------
    n_el = 40
    base = Beam(z_f0, z_top, n_el, lambda s: (1.0, 1.0, 1.0))
    feed = Beam(z_f0, z_top, n_el, lambda s: (1.0, 1.0 if z_ft <= s <= z_xh0 else 0.0, 1.0))
    st_sets = {1: [(z_xh0 + t_xh / 2, "pin")], 2: [(z_xh0 + t_xh / 2, "pin"), (zc_stow + 0.5 * env_h, "pin")]}
    # unit-EI static responses (deflection + slope) for each latch height, plane, pose and load
    unit_resp: Dict[Tuple[float, str], Dict[str, Any]] = {}
    for h_l in mc["latch_height_options_m"]:
        z_l = z_h + float(h_l)
        sup = {"fore_aft": [(z_h, "pin"), (z_l, "pin")], "side": [(z_h, "clamp"), (z_l, "pin")]}
        for pl, sp in sup.items():
            resp = []
            for zc in drill_poses:
                cases = {"m_top": [(z_top, 0.0, 1.0)], "m_car": [(zc, 0.0, 1.0)], "f_car": [(zc, 1.0, 0.0)], "f_bush": [(z_bm, 1.0, 0.0)]}
                r = {}
                for k, ld in cases.items():
                    st = base.static(sp, loads=ld)
                    nc, nb = base.node(zc), base.node(z_bm)
                    r[k] = (st["w"][nc] + st["phi"][nc] * (zc - z_bm) - st["w"][nb])   # offset at the bushing per unit load / EI
                resp.append(r)
            unit_resp[(float(h_l), pl)] = {"sup": sp, "resp": resp}
    cands: List[Dict[str, Any]] = []
    for mname in mc["material_options"]:
        mat = mats[mname]
        e_m, rho_m = float(mat["youngs_modulus_pa"]), float(mat["density_kg_m3"])
        for b_mm, h_mm, t_mm in mc["section_options_mm"]:
            b, h, t = b_mm / 1e3, h_mm / 1e3, t_mm / 1e3
            sec = box_section(b, h, t)
            mu_sp = rho_m * sec["A"] * mga
            pm = plate_masses(h)
            y_sc = y_sf + h / 2
            ei = {"fore_aft": e_m * sec["Ix"], "side": e_m * sec["Iy"]}
            st_cache: Dict[int, Dict[int, float]] = {}

            def lumps_at(zc: float) -> List[Tuple[float, float]]:
                return [(zc + 0.5 * env_h, m_mov), (z_top, pm["crosshead"]), (z_f0, pm["foot"])]

            def mass_matrix(n: int) -> Tuple[float, np.ndarray]:
                mu_f = (mu_rails + mu_screw) * mga + n * m_bracket / l_rail
                return mu_f, mu_sp * base.M + mu_f * feed.M
            for h_l in mc["latch_height_options_m"]:
                z_l = z_h + float(h_l)

                def offset(n: int) -> float:           # worst string offset at the bushing over the drilling stroke
                    worst = 0.0
                    for k, zc in enumerate(drill_poses):
                        side = (r_carr, r_bush) if k == 0 else (0.0, 0.0)
                        ra = unit_resp[(float(h_l), "fore_aft")]["resp"][k]
                        rs = unit_resp[(float(h_l), "side")]["resp"][k]
                        d_fa = (wob_lim * (y_ls - y_sc) * ra["m_top"] - wob_lim * y_ls * ra["m_car"] - side[0] * ra["f_car"]
                                + side[1] * ra["f_bush"]) / ei["fore_aft"] + rail_tilt(n) * (zc - z_bm)
                        d_sd = (-side[0] * rs["f_car"] + side[1] * rs["f_bush"]) / ei["side"]
                        worst = max(worst, math.hypot(d_fa, d_sd))
                    return worst
                # fewest brackets that meet the alignment budget (more brackets only add mass)
                n_br = next((n for n in range(n_br_min, int(mc["max_rail_brackets"]) + 1) if offset(n) <= clr), int(mc["max_rail_brackets"]))
                worst = offset(n_br)
                mu_feed, M = mass_matrix(n_br)
                if n_br not in st_cache:
                    st_cache[n_br] = {n: min(base.f1([(z_h, "pin")] + hds, lumps_at(zc_stow), ei["fore_aft"], M),
                                             base.f1([(z_h, "clamp")] + hds, lumps_at(zc_stow), ei["side"], M))
                                      for n, hds in st_sets.items()}
                f_dep = [min(base.f1(unit_resp[(float(h_l), pl)]["sup"], lumps_at(zc), ei[pl], M) for pl in ei) for zc in drill_poses]
                ok_f = not any(in_band(f, forcing, sep) for f in f_dep)
                l_c = z_top - z_l
                p_cr = math.pi ** 2 * min(ei.values()) / (2 * l_c) ** 2
                p_comp = pull + m_head * g_max
                m_spine = mu_sp * (z_top - z_f0)
                m_posts = rho_m * sec["A"] * mga * 2 * float(h_l) * (1 + 2 * math.sqrt(2))
                cands.append({"material": mname, "b": b, "h": h, "t": t, "h_l": float(h_l), "n_br": n_br, "sec": sec, "E": e_m,
                              "mu_sp": mu_sp, "mu_feed": mu_feed, "m_spine": m_spine, "m_posts": m_posts,
                              "m_obj": m_spine + m_posts + n_br * m_bracket, "f_dep": f_dep, "f_stow": st_cache[n_br],
                              "align_m": worst, "tilt": rail_tilt(n_br), "p_cr": p_cr, "p_comp": p_comp,
                              "ok_dep": ok_f and worst <= clr and p_cr >= fos_u * p_comp, "plates": pm})
    m_hd = float(mc["hold_down_unit_mass_kg"])
    ok = [dict(c, n_hd=n) for c in cands for n in (1, 2) if c["ok_dep"] and c["f_stow"][n] >= f_req]
    chosen = min(ok, key=lambda c: c["m_obj"] + c["n_hd"] * m_hd) if ok else None
    assert chosen is not None, "no mast section / latch height / bracket count passes the deployed and stowed checks"
    c = chosen
    n_feasible = sum(1 for x in cands if x["ok_dep"])
    b, h, t, h_l, n_br = c["b"], c["h"], c["t"], c["h_l"], c["n_br"]
    sec, e_m = c["sec"], c["E"]
    z_l = z_h + h_l
    y_sc, y_h = y_sf + h / 2, y_sf + h
    mat = mats[c["material"]]
    mu_sp, mu_feed, pm = c["mu_sp"], c["mu_feed"], c["plates"]
    M = mu_sp * base.M + mu_feed * feed.M
    label = "6061-T6 Al" if c["material"] == "mast_aluminium" else "Ti-6Al-4V"
    led.add("mast_section", f"{b*1e3:.0f}x{h*1e3:.0f}x{t*1e3:.0f} {label}", "mm", "min (m_spine + m_posts + m_brackets) over feasible candidates",
            f"{n_feasible} of {len(cands)} (material, section, brackets, latch height) candidates pass the deployed checks")
    led.add("mast_latch_height_mm", round(h_l * 1e3, 0), "mm", "latch posts: chosen with the section",
            f"latch at z = {z_l*1e3:.0f} mm; {(z_top - z_l)*1e3:.0f} mm of spine above it")
    led.add("rail_bracket_count", n_br, "-", "spans <= L_max and the alignment budget",
            f"{n_br} brackets, {l_rail/(n_br+1)*1e3:.0f} mm spans ({n_br_min} needed for the launch mode alone); "
            f"rail flex tilts the carriage {c['tilt']*1e6:.0f} urad")
    led.add("mast_spine_mass_kg", round(c["m_spine"], 2), "kg", "m = rho A L (1 + MGA)",
            f"A = {sec['A']*1e6:.0f} mm^2 over {(z_top - z_f0)*1e3:.0f} mm; MGA {mc['mass_growth_allowance']:.0%} (AIAA S-120A)")
    f_dep = c["f_dep"]
    lumps_home = [(zc_home + 0.5 * env_h, m_mov), (z_top, pm["crosshead"]), (z_f0, pm["foot"])]
    sup_dep = {"fore_aft": [(z_h, "pin"), (z_l, "pin")], "side": [(z_h, "clamp"), (z_l, "pin")]}
    ei = {"fore_aft": e_m * sec["Ix"], "side": e_m * sec["Iy"]}
    f_home = min(base.f1(sup_dep[pl], lumps_home, ei[pl], M) for pl in ei)
    edges = sorted([lo / (1 + sep) for lo, _ in forcing] + [hi * (1 + sep) for _, hi in forcing])
    band_margin = min(abs(f - e) / e for f in f_dep for e in edges)
    led.add("mast_deployed_f1_hz", round(min(f_dep), 1), "Hz", "K phi = w^2 M phi: hinge + latch, carriage over the drilling stroke",
            f"{min(f_dep):.1f}-{max(f_dep):.1f} Hz from the start of the hole to 1 m; outside the rotary {forcing[0][0]:.0f}-{forcing[0][1]:.0f} Hz, "
            f"cam {forcing[1][0]:.0f} Hz and blow {forcing[2][0]:.0f} Hz bands (+/-{sep:.0%}); {band_margin:.0%} from the nearest band edge")
    led.add("mast_home_f1_hz", round(f_home, 1), "Hz", "same model, carriage at home (no percussion there)")
    led.add("mast_stowed_f1_hz", round(c["f_stow"][c["n_hd"]], 1), "Hz", "hinge + hold-down(s), carriage at the stow pose",
            f"requirement {f_req:.0f} Hz; {c['n_hd']} hold-down(s) (one at the crosshead alone gives {c['f_stow'][1]:.1f} Hz)")
    led.add("mast_alignment_offset_mm", round(c["align_m"] * 1e3, 3), "mm", "e = w_c + phi_c L_cb - w_b at WOB_lim (+ side force)",
            f"the string leaves the carriage {c['align_m']*1e3:.3f} mm off the bushing axis vs the {clr*1e3:.1f} mm bushing clearance")
    theta = c["align_m"] / (drill_poses[-1] - z_bm)
    led.add("bit_alignment_error_1m_mm", round(theta * (z_bm + target) * 1e3, 3), "mm", "e = theta (z_bush + 1 m), theta = offset / L_cb",
            f"string tilt {math.degrees(theta)*3600:.0f} arc-s about the bushing at full depth (hole-wall guidance ignored)")
    led.add("mast_buckling_load_n", round(c["p_cr"], 0), "N", "P_cr = pi^2 EI_min / (2 L)^2 above the latch",
            f"the spine is in tension while drilling; compression only when pulling a stuck string ({c['p_comp']:.0f} N): FOS {c['p_cr']/c['p_comp']:.0f}")

    # ---- Launch: spine stress, hinge and hold-down reactions ----------------------------------------
    lumps_st = [(zc_stow + 0.5 * env_h, m_mov), (z_top, pm["crosshead"]), (z_f0, pm["foot"])]
    hds = st_sets[c["n_hd"]]
    m_rot = mu_sp * (z_top - z_f0) + mu_feed * l_rail + sum(m for _, m in lumps_st)
    launch = {}
    for pl, (ei_p, c_p, sup) in {"deck_normal": (ei["fore_aft"], h / 2, [(z_h, "pin")] + hds),
                                 "lateral": (ei["side"], b / 2, [(z_h, "clamp")] + hds)}.items():
        f_p = base.f1(sup, lumps_st, ei_p, M)
        g_eq = max(qs, miles_3sigma_g(p, f_p))
        bk = Beam(z_f0, z_top, n_el, lambda s, e=ei_p, cc=c_p: (e, mu_sp + (mu_feed if z_ft <= s <= z_xh0 else 0.0), cc))
        r = bk.static(sup, lumps_st, accel=g_eq * G0)
        n_h = 2 * bk.node(z_h)
        launch[pl] = {"f1_hz": f_p, "g": g_eq, "sigma_mpa": r["sig_over_e"] * e_m / 1e6, "hinge_n": abs(r["react"][n_h]),
                      "hinge_moment_nm": abs(r["react"].get(n_h + 1, 0.0)),
                      "hold_down_n": max(abs(r["react"][2 * bk.node(s)]) for s, _ in hds)}
    sig_l = max(v["sigma_mpa"] for v in launch.values()) * 1e6
    g_ax = max(v["g"] for v in launch.values())
    led.add("mast_launch_stress_mpa", round(sig_l / 1e6, 1), "MPa", "sigma = M c / I, launch g = max(QS, Miles 3-sigma)",
            f"yield FOS {float(mat['yield_strength_pa'])/sig_l:.1f} (>= {fos_y}), ultimate FOS {float(mat['ultimate_strength_pa'])/sig_l:.1f} (>= {fos_u})")
    led.add("mast_rotating_mass_kg", round(m_rot, 2), "kg", "spine + rails + brackets + screw + carriage + crosshead + foot",
            f"incl. the {m_mov} kg carriage and string; MGA {mc['mass_growth_allowance']:.0%}")
    t_lug = float(mc["lug_thickness_m"])
    lug_sp = b + t_lug
    f_hinge_lug = max(launch["deck_normal"]["hinge_n"] / 2,
                      launch["lateral"]["hinge_n"] / 2 + launch["lateral"]["hinge_moment_nm"] / lug_sp,
                      m_rot * g_ax * G0 / 2)
    hp = pick_pin(p, f_hinge_lug, t_lug)
    led.add("hinge_lug_load_n", round(f_hinge_lug, 0), "N", "max(R_n/2, R_lat/2 + M/s, m g_ax / 2) at launch",
            f"two Ti lugs {lug_sp*1e3:.0f} mm apart; the hinge also takes the launch load along the stowed mast")
    led.add("hinge_pin_d_mm", round(hp["d_m"] * 1e3, 0), "mm", "min d: tau FOS_u <= S_u/sqrt3, bearing FOS_y <= S_y,lug",
            f"17-4PH pin in double shear {hp['tau_mpa']:.0f} MPa (FOS {hp['fos_shear']:.1f}), lug bearing {hp['bearing_mpa']:.0f} MPa")
    # Hold-down: SMA-released frangible bolt in a cup-cone (cone takes shear, bolt takes tension)
    p_t = max(v["hold_down_n"] for v in launch.values())
    bolt = mats["bolt_a286"]
    pre_f = float(mc["hold_down_preload_fraction"])
    sep_f = float(cr["joint_separation_fos"])
    hd_pick = None
    for name, a_s in mc["hold_down_bolts"].items():
        p_pre = pre_f * float(bolt["yield_strength_pa"]) * a_s
        if p_pre >= sep_f * fos_u * p_t and p_pre + fos_u * p_t <= float(bolt["ultimate_strength_pa"]) * a_s:
            hd_pick = (name, float(a_s), p_pre)
            break
    assert hd_pick is not None, "no hold-down bolt passes"
    led.add("hold_down_tension_n", round(p_t, 0), "N", "hold-down reaction, launch g = max(QS, Miles 3-sigma)",
            f"{c['n_hd']} hold-down(s); the hinge is the other support")
    led.add("hold_down_bolt", hd_pick[0], "-", "min A_s: P_pre >= FS_sep FOS_u P_t and P_pre + FOS_u P_t <= S_u A_s",
            f"A-286 frangible bolt, preload {hd_pick[2]:.0f} N ({pre_f:.0%} of yield); SMA release (Frangibolt class), no pyro shock")

    # ---- Deployment actuator and latch -------------------------------------------------------------
    members = [  # (mass per length, y, z0, z1) and point masses (m, y, z), deployed frame, home pose
        (mu_sp, y_sc, z_f0, z_top), (mu_rails * mga, 0.0, z_ft, z_xh0), (n_br * m_bracket / l_rail, 0.5 * (r_rail + y_sf), z_ft, z_xh0),
        (mu_screw * mga, y_ls, z_ft, z_xh0)]
    points = [(m_head, 0.02, zc_home + 0.5 * env_h), (m_str, 0.0, zc_home - l_aug / 2),
              (pm["crosshead"], 0.5 * (y_h - r_lb), z_top + 0.5 * l_mot), (pm["foot"], pm["y_foot"], z_f0)]
    mass = sum(mu * (z1 - z0) for mu, _, z0, z1 in members) + sum(m for m, _, _ in points)
    dy = (sum(mu * (z1 - z0) * y for mu, y, z0, z1 in members) + sum(m * y for m, y, _ in points)) / mass - y_h
    dz = (sum(mu * (z1 ** 2 - z0 ** 2) / 2 for mu, _, z0, z1 in members) + sum(m * z for m, _, z in points)) / mass - z_h
    j_h = (sum(mu * ((y - y_h) ** 2 * (z1 - z0) + ((z1 - z_h) ** 3 - (z0 - z_h) ** 3) / 3) for mu, y, z0, z1 in members)
           + sum(m * ((y - y_h) ** 2 + (z - z_h) ** 2) for m, y, z in points))
    th = np.radians(np.linspace(0.0, 90.0, 91))
    arm = float(np.max(np.abs(dy * np.cos(th) + dz * np.sin(th))))   # horizontal CG offset from the hinge while deploying
    t_grav = mass * g_max * arm
    t_fric = float(mc["hinge_friction_coeff"]) * mass * g_max * hp["d_m"] / 2
    t_harn = float(mc["harness_torque_nm"])
    t_dep, f_acc = float(mc["deploy_time_s"]), float(mc["deploy_accel_fraction"])
    w_max = (math.pi / 2) / (t_dep * (1 - f_acc))
    t_in = j_h * w_max / (f_acc * t_dep)
    t_req = float(cr["motorisation_static_factor"]) * (t_grav + t_fric + t_harn) + float(cr["motorisation_inertial_factor"]) * t_in
    n_g, eta_g = float(mc["actuator_gear_ratio"]), float(mc["actuator_gear_efficiency"])
    t_m = t_req / (n_g * eta_g)
    d_rot = (4 * t_m / (math.pi * float(hd["motor_trv_n_m_m3"]) * float(hd["motor_rotor_aspect"]))) ** (1 / 3)
    led.add("deploy_gravity_torque_nm", round(t_grav, 2), "N m", "T_g = m g_Mars max|dy cos th + dz sin th|",
            f"rotating mass {mass:.2f} kg, CG {dz*1e3:.0f} mm above / {-dy*1e3:.0f} mm fore of the hinge (home pose)")
    led.add("deploy_actuator_torque_nm", round(t_req, 2), "N m", "T = 2 (T_g + T_f + T_harness) + 1.25 J alpha",
            f"friction {t_fric:.3f}, harness {t_harn}, inertia {t_in*1e3:.2f} mN m (J = {j_h:.2f} kg m^2, {t_dep:.0f} s); ECSS-E-ST-33-01C")
    led.add("deploy_motor_rotor_d_mm", round(d_rot * 1e3, 1), "mm", "D_r = (4 T_m / (pi TRV lambda))^(1/3), T_m = T / (i eta)",
            f"{n_g:.0f}:1 strain-wave gear, eta {eta_g}: motor {t_m*1e3:.0f} mN m. Earth tests need a g-offload "
            f"({mass*G0*arm:.0f} N m at 1 g)")
    lat = 0.0
    for k, zc in enumerate(drill_poses):
        side = (r_carr, r_bush) if k == 0 else (0.0, 0.0)
        bk = Beam(z_f0, z_top, n_el, lambda s: (ei["fore_aft"], 0.0, h / 2))
        r = bk.static(sup_dep["fore_aft"], loads=[(z_top, 0.0, wob_lim * (y_ls - y_sc)), (zc, 0.0, -wob_lim * y_ls),
                                                  (zc, -side[0], 0.0), (z_bm, side[1], 0.0)])
        lat = max(lat, abs(r["react"][2 * bk.node(z_l)]))
    f_latch = (lat + mass * g_max * abs(dy) / h_l) / 2
    lp = pick_pin(p, f_latch, t_lug)
    led.add("latch_pin_load_n", round(f_latch, 0), "N", "(R_latch,WOB + m g |dy| / h_l) / 2 posts",
            f"latch pin {lp['d_m']*1e3:.0f} mm, shear FOS {lp['fos_shear']:.0f}: stiffness, not strength, sizes the latch")
    e_kin = 0.5 * j_h * w_max ** 2
    led.add("latch_engagement_energy_mj", round(e_kin * 1e3, 3), "mJ", "E = J w^2 / 2 at the end of the slew",
            "absorbed by the spring latch pin and a Vespel stop; the actuator brakes over the last 10 % anyway")

    # ---- Leadscrew: buckling and whirl ---------------------------------------------------------------
    l_buck = z_xh0 - (drill_poses[-1] + nut_off)
    p_cr_ls = math.pi ** 2 * e_fs * i3 / (0.7 * l_buck) ** 2
    led.add("leadscrew_buckling_load_n", round(p_cr_ls, 0), "N", "P_cr = pi^2 E I_3 / (0.7 L)^2 (fixed top bearing, nut pinned)",
            f"nut-to-top length {l_buck*1e3:.0f} mm at 1 m depth; FOS {p_cr_ls/wob_lim:.1f} on the {wob_lim:.0f} N lift-off WOB")
    n_cr = 60 / (2 * math.pi) * (3.927 / l_rail) ** 2 * math.sqrt(e_fs * i3 / mu_screw)
    n_max = float(zf["retraction_speed_m_s"]) / lead * 60
    led.add("leadscrew_critical_speed_rpm", round(n_cr, 0), "rpm", "n_cr = (60/2pi)(3.927/L)^2 sqrt(EI/rho A) (fixed-supported)",
            f"{l_rail*1e3:.0f} mm between bearings; retraction {n_max:.0f} rpm = {n_max/n_cr:.0%} of critical (limit {cr['leadscrew_speed_fraction']:.0%})")
    l_nut = head["nut"]["length_m"]
    p_nut = m_mov * g_ax * G0 / (math.pi * d2 * 0.5 * lead * (l_nut / lead))
    led.add("leadscrew_nut_launch_pressure_mpa", round(p_nut / 1e6, 2), "MPa", "p = m_c g_launch / (pi d_2 h_1 n_t)",
            f"launch along the stowed mast: the self-locking nut holds the carriage; allowable {float(hd['nut_allowable_pressure_pa'])/1e6:.0f} MPa")
    out["leadscrew"] = {"d3_m": d3, "p_cr_n": p_cr_ls, "n_cr_rpm": n_cr, "n_max_rpm": n_max, "nut_launch_pa": p_nut,
                        "l_buck_m": l_buck, "span_m": l_rail}

    # ---- Lander deck, centre of mass, legs, tilt and torque reaction --------------------------------
    w_deck = float(li["deck_width_m"])
    l_deck = math.ceil((z_top - z_h + 0.04) / 0.05) * 0.05     # hold-down pedestal under the crosshead fits on the deck
    m_lander = float(p["system"]["lander_mass_kg"])
    m_drill = mass + c["m_posts"]
    m_rest = m_lander - m_drill
    assert m_rest > 0
    y_drill = (mass * (y_h + dy) + c["m_posts"] * y_h) / m_drill
    y_g = (m_rest * (y_h + l_deck / 2) + m_drill * y_drill) / m_lander
    wob_set_min = min(w["wob_setpoint_n"] for w in wob.values())
    fos_w = float(cr["wob_lift_off_fos"])
    k_tilt = 1.0 + float(head["load_cell_error_n"]) / wob_set_min
    a_min = y_g * k_tilt / (fos_w - k_tilt)
    splay = float(li["leg_splay_m"])
    y_far = max(y_h + l_deck + splay, y_g + a_min)
    y_near = y_h - splay
    ratio = (y_far - y_g) / (y_far - 0.0)
    led.add("lander_deck_length_m", l_deck, "m", "L >= z_top - z_hinge + hold-down pedestal",
            f"the stowed mast lies aft over the deck; {w_deck:.2f} m wide, top {z_deck:.2f} m above the regolith")
    led.add("lander_cg_offset_m", round(y_g, 3), "m", "y_G = sum m_i y_i / m (deployed)",
            f"drill {m_drill:.1f} kg at y = {y_drill:.2f} m, rest of lander {m_rest:.1f} kg at the deck centre")
    led.add("lander_tilt_lift_off_ratio", round(ratio, 3), "-", "WOB_tip / (m g) = a / (a + e)",
            f"a side-mounted drill unloads the near legs before the lander rises: tilt about the far legs {y_far - y_g:.2f} m behind the CG")
    led.add("lander_far_leg_line_m", round(y_far, 3), "m", "max(deck end + splay, y_G + e k / (FOS - k)), k = 1 + e_LC / WOB_set",
            f"WOB_set = m g / {fos_w} is kept; near legs stay loaded at the set-point plus the load-cell error. "
            f"Tilt margin {ratio*fos_w:.2f} vs {fos_w} for a drill through the CG")
    wobs = {}
    for k, g in g_env.items():
        tip_n = m_lander * g * ratio
        n_near, n_far = leg_reactions(m_lander * g, wob[k]["wob_setpoint_n"], y_g, y_near, y_far)
        pads = [(w_deck / 2 + splay, y_near, n_near), (w_deck / 2 + splay, y_far, n_far)]
        t_cap = 2 * float(li["footpad_friction_coeff"]) * sum(n * math.hypot(x, y) for x, y, n in pads)
        wobs[k] = {"wob_tip_n": tip_n, "margin": tip_n / wob[k]["wob_setpoint_n"], "near_leg_n": n_near, "far_leg_n": n_far,
                   "torque_capacity_nm": t_cap, "torque_fos": t_cap / float(rot["max_safe_torque_nm"])}
        led.add(f"wob_tilt_limit_{k}_n", round(tip_n, 2), "N", "WOB_tip = m g a / (a + e)",
                f"set-point {wob[k]['wob_setpoint_n']} N (margin {tip_n/wob[k]['wob_setpoint_n']:.2f}); near legs {n_near:.1f} N, far {n_far:.1f} N each")
        led.add(f"lander_torque_reaction_fos_{k}", round(t_cap / float(rot["max_safe_torque_nm"]), 2), "-",
                "mu sum N_i r_i / T_max (footpads about the drill axis)",
                f"footpad friction alone reacts the {rot['max_safe_torque_nm']} N m torque limit at the WOB set-point")
    t_fos_min = min(v["torque_fos"] for v in wobs.values())
    assert t_fos_min >= fos_w, f"footpad friction cannot react the drilling torque (FOS {t_fos_min:.2f}): anchors needed"

    # ---- Solar wings: clear the legs at the steepest tilt, at the lander CG station --------------------
    sa = p["power_system"]["solar_array"]
    ww, wh = float(sa["wing_width_m"]), float(sa["wing_height_m"])
    beta = max_sun_elevation_deg(p)
    r_corner = math.hypot(ww / 2, wh / 2 * math.sin(math.radians(beta)))
    c_w = float(sa["wing_deck_clearance_m"])
    x_w = w_deck / 2 + splay + r_corner + c_w
    z_w0 = z_deck + 0.022 + c_w                                 # lower panel edge above the azimuth platter
    led.add("wing_axis_x_m", round(x_w, 3), "m", "x_w = W/2 + splay + sqrt((w/2)^2 + (h/2 sin beta)^2) + c",
            f"steepest tracking tilt {beta:.0f} deg (Mars, Ls 90): the wing corner sweeps r = {r_corner:.2f} m and clears the legs by {c_w*100:.0f} cm")
    led.add("wing_base_height_m", round(z_w0, 3), "m", "z = deck top + azimuth platter + c",
            "wings ride on azimuth turntables at the ends of deck outriggers, at the lander CG station")

    # ---- Envelopes for the solar shading model and the CAD (m) ---------------------------------------
    x_post = max(b / 2 + t_lug, half_w) + c_run + 0.02        # latch post centreline (20 mm half-width) beside the swept mast
    z_mtop = z_top + l_mot
    out["solar"] = {
        "wing_x_m": x_w, "wing_y_m": y_g, "wing_z0_m": z_w0,
        "boxes_m": [
            [[-half_w, -0.060, 0.0], [half_w, y_h, z_mtop]],                                 # mast + carriage envelope
            [[-w_deck / 2, y_h, z_deck - float(li["deck_thickness_m"])], [w_deck / 2, y_h + l_deck, z_deck + 0.28]],  # deck + tanks
            [[-x_post - 0.02, y_sf, z_deck], [x_post + 0.02, y_h + h_l, z_l + 0.03]],        # latch posts
        ],
    }
    out["geometry_m"] = {
        "rail_x": rail_x, "y_ls": y_ls, "y_bracket": y_br0, "y_spine_fore": y_sf, "y_hinge": y_h, "z_hinge": z_h, "z_deck": z_deck,
        "z_latch": z_l, "z_foot_top": z_ft, "z_foot_bottom": z_f0, "z_crosshead_bottom": z_xh0, "z_top": z_top,
        "z_motor_top": z_mtop, "half_width": half_w, "y_foot_fore": y_foot_fore, "foot_disc_r": r_disc, "x_post": x_post,
        "b": b, "h": h, "t": t, "bracket_neck": t_neck, "bracket_t": t_br, "bracket_depth": d_br, "n_brackets": n_br,
        "crosshead_t": t_xh, "foot_t": t_ft, "motor_d": d_mot, "motor_l": l_mot, "lug_t": t_lug, "lug_spacing": lug_sp,
        "hinge_pin_d": hp["d_m"], "latch_pin_d": lp["d_m"], "deck_length": l_deck, "deck_width": w_deck,
        "deck_t": float(li["deck_thickness_m"]), "y_leg_near": y_near, "y_leg_far": y_far, "x_leg": w_deck / 2 + splay,
        "footpad_r": float(li["footpad_radius_m"]), "stow_gap": float(mc["stow_gap_m"]), "hold_downs_z": [s for s, _ in hds],
        "actuator_rotor_d": d_rot, "actuator_stator_od": d_rot / float(hd["motor_split_ratio"]), "y_cg": y_g,
        "y_stow": [y_h + (s - z_h) for s, _ in hds], "carriage_drop": d_stow, "wing_x": x_w, "wing_z0": z_w0,
        "brush_stow_deg": float(brush["arm"]["stow_deg"]),
        "bolt_d": math.sqrt(4 * hd_pick[1] / math.pi) / 0.9,
    }
    out.update({
        "material": c["material"], "label": label, "section_mm": [b * 1e3, h * 1e3, t * 1e3], "latch_height_m": h_l,
        "n_brackets": n_br, "n_hold_downs": c["n_hd"], "hold_down_bolt": hd_pick[0], "hold_down_tension_n": p_t,
        "f_deployed_hz": f_dep, "f_home_hz": f_home, "f_stowed_hz": c["f_stow"][c["n_hd"]], "f_stowed_1hd_hz": c["f_stow"][1],
        "f_required_hz": f_req, "forcing_hz": forcing, "band_margin": band_margin, "separation": sep, "alignment_m": c["align_m"], "clearance_m": clr,
        "rail_tilt_rad": c["tilt"], "rail_span_max_m": l_span_max, "rail_f_end_hz": f_end, "rail_span_m": l_rail / (n_br + 1),
        "buckling_n": c["p_cr"], "compression_n": c["p_comp"], "launch": launch, "launch_stress_mpa": sig_l / 1e6,
        "yield_pa": float(mat["yield_strength_pa"]), "ultimate_pa": float(mat["ultimate_strength_pa"]),
        "rotating_mass_kg": mass, "spine_mass_kg": c["m_spine"], "posts_mass_kg": c["m_posts"],
        "hinge_pin": hp, "latch_pin": lp, "hinge_lug_n": f_hinge_lug, "latch_n": f_latch,
        "deploy": {"t_grav_nm": t_grav, "t_fric_nm": t_fric, "t_harness_nm": t_harn, "t_inertia_nm": t_in, "t_req_nm": t_req,
                   "j_kg_m2": j_h, "cg_dy_m": dy, "cg_dz_m": dz, "arm_m": arm, "motor_torque_nm": t_m, "rotor_d_m": d_rot,
                   "mass_kg": mass},
        "lander": {"deck_length_m": l_deck, "cg_y_m": y_g, "far_leg_y_m": y_far, "near_leg_y_m": y_near,
                   "tilt_ratio": ratio, "k_tilt": k_tilt, "a_min_m": a_min, "m_rest_kg": m_rest, "m_drill_kg": m_drill, "env": wobs},
        "side_force_n": f_side, "side_carriage_n": r_carr, "side_bushing_n": r_bush, "wob_design_n": wob_lim,
        "candidates_feasible": n_feasible, "candidates_total": len(cands),
    })
    return out
