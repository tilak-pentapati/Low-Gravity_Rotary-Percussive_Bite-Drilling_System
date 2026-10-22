"""
Enclosed rotary-percussive drill head - detailed build123d geometry (mm, Z-up, spindle on the Z axis).

Every size comes from config/system_params.yaml (drill_head, actuators) or from src/design_head.py
(motor diameters, gear modules, spring, cam profile, bearings, axial stack, plan-view offsets).

Load paths
  WOB      : leadscrew nut -> top yoke -> load cell -> housing -> 7004 pair -> spindle end face -> anvil web
             -> auger hex shank.  (The housing hangs from the load cell; a diaphragm flexure centres its nose.)
  Impact   : hammer -> anvil strike face -> hex shank -> bit.  The anvil slides on the spindle spline, so the
             stress wave never crosses the bearings.
  Torque   : BLDC -> 8:1 planetary -> 12T pinion -> 24T spindle gear -> spline -> anvil -> hex.
  Percussion: secondary BLDC -> 18T pinion -> 54T ring gear on the barrel cam (61812 bearing) -> two
             cycloidal lobes lift the hammer rollers -> spring -> hammer drops onto the anvil.

Returns four part groups: carriage (static), spindle (turns with the auger), hammer (hammer + follower
rollers, axial stroke) and cam (barrel cam ring).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import build123d as bd

ALIGN_BASE = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
TAN20 = math.tan(math.radians(20.0))
PartT = Tuple[str, bd.Shape, str]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _cyl(r, h, z0, x=0.0, y=0.0):
    return bd.Pos(x, y, z0) * bd.Cylinder(r, h, align=ALIGN_BASE)


def _tube(ro, ri, h, z0, x=0.0, y=0.0):
    return _cyl(ro, h, z0, x, y) - _cyl(ri, h + 0.2, z0 - 0.1, x, y)


def _toothed(z: int, r_tip: float, r_root: float, r_ref: float) -> List[Tuple[float, float]]:
    """Closed outline of z straight-flank (20 deg) teeth: tips at r_tip, roots at r_root (either order)."""
    tau = 2 * math.pi / z
    w_ref = tau / 4
    w_tip = w_ref - abs(r_tip - r_ref) * TAN20 / r_tip
    w_root = min(w_ref + abs(r_ref - r_root) * TAN20 / r_root, 0.48 * tau)
    pts = []
    for i in range(z):
        c = i * tau
        for r, a in ((r_root, c - 0.5 * tau), (r_root, c - w_root), (r_tip, c - w_tip),
                     (r_tip, c + w_tip), (r_root, c + w_root)):
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _gear(m: float, z: int, h: float, z0: float, x=0.0, y=0.0, bore=0.0, rot=0.0) -> bd.Part:
    """External spur gear (straight-flank approximation of the involute), chamfered tips."""
    r = m * z / 2
    g = bd.extrude(bd.Polygon(*_toothed(z, r + m, r - 1.25 * m, r), align=None), amount=h)
    if bore > 0:
        g = g - _cyl(bore, h + 1, -0.5)
    return bd.Pos(x, y, z0) * bd.Rot(0, 0, rot) * g


def _ring_gear(m: float, z: int, h: float, z0: float, r_out: float, x=0.0, y=0.0) -> bd.Part:
    r = m * z / 2
    cut = bd.extrude(bd.Polygon(*_toothed(z, r - m, r + 1.25 * m, r), align=None), amount=h + 1)
    return bd.Pos(x, y, z0) * (_cyl(r_out, h, 0) - bd.Pos(0, 0, -0.5) * cut)


def _spline(z: int, r_major: float, r_root: float, r_ref: float, h: float, z0: float) -> bd.Part:
    return bd.Pos(0, 0, z0) * bd.extrude(bd.Polygon(*_toothed(z, r_major, r_root, r_ref), align=None), amount=h)


def _bearing(name: str, d: float, D: float, B: float, z0: float, x=0.0, y=0.0) -> List[PartT]:
    """Rings (with land between) + ball set for a catalogue ball bearing."""
    t = (D - d) / 2
    rings = _tube(D / 2, D / 2 - 0.3 * t, B, z0, x, y) + _tube(d / 2 + 0.3 * t, d / 2, B, z0, x, y)
    rp, rb = (d + D) / 4, 0.42 * t / 2 * 1.6
    n = int(math.pi * 2 * rp / (2.3 * rb))
    balls = [bd.Pos(x + rp * math.cos(2 * math.pi * i / n), y + rp * math.sin(2 * math.pi * i / n), z0 + B / 2)
             * bd.Sphere(rb) for i in range(n)]
    return [(f"{name}_Rings", rings, "gearbox_gunmetal"), (f"{name}_Balls", bd.Compound(children=balls), "steel_shaft")]


def _coil(coil_r: float, wire_r: float, z0: float, length: float, turns: float) -> bd.Part:
    """Compression spring, squared and ground: helix swept, then ground flat to the free length."""
    helix = bd.Helix(pitch=(length - 2 * wire_r) / turns, height=length - 2 * wire_r, radius=coil_r)
    plane = bd.Plane(origin=helix @ 0, x_dir=(1, 0, 0), z_dir=helix % 0)
    with bd.BuildPart() as bp:
        with bd.BuildLine():
            bd.add(helix)
        with bd.BuildSketch(plane):
            bd.Circle(wire_r)
        bd.sweep(is_frenet=True)
    s = bd.Pos(0, 0, z0 + wire_r) * bp.part
    return s & _cyl(coil_r + 2 * wire_r, length - 0.6 * wire_r, z0 + 0.3 * wire_r)


def _motor(tag: str, x: float, y: float, z0: float, d_s: float, d_r: float, l: float) -> List[PartT]:
    """Frameless BLDC: stator (laminations), copper end turns, PM rotor, shaft and encoder disc."""
    rs, rr = d_s / 2, d_r / 2
    parts: List[PartT] = [
        (f"{tag}_BLDC_Stator", _tube(rs, rr + 0.4, l, z0 + 6, x, y), "bldc_emerald"),
        (f"{tag}_BLDC_End_Turns", _tube(rs - 2.5, rr + 2.0, 5.5, z0, x, y) + _tube(rs - 2.5, rr + 2.0, 5.5, z0 + l + 6.5, x, y),
         "copper_winding"),
        (f"{tag}_BLDC_PM_Rotor", _tube(rr, rr - 3.0, l, z0 + 6, x, y), "tungsten_carbide"),
        (f"{tag}_BLDC_Shaft", _cyl(min(4.0, 0.25 * rr), l + 36, z0 - 20, x, y), "steel_shaft"),
        (f"{tag}_Encoder_Disc", _cyl(rr + 1.0, 1.0, z0 + l + 14, x, y), "gold_kapton"),
    ]
    return parts


def _cam_track(prof: List[Tuple[float, float]], lobes: int, r_in: float, r_out: float, z0: float, step: float = 2.5) -> bd.Part:
    """Barrel-cam track: annulus whose top follows the generated lift s(theta) (stepped every `step` deg)."""
    period = 360.0 / lobes
    lift = dict(prof)
    keys = sorted(lift)
    solids = []
    for lobe in range(lobes):
        th = 0.0
        while th < period - 1e-9:
            th1 = min(th + step, period)
            mid = 0.5 * (th + th1)
            s = max(lift[min(keys, key=lambda k: abs(k - mid))], 0.0)
            if s > 0.02:
                a0, a1 = math.radians(th + lobe * period), math.radians(th1 + lobe * period)
                n = 3
                outer = [(r_out * math.cos(a0 + (a1 - a0) * i / n), r_out * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
                inner = [(r_in * math.cos(a1 - (a1 - a0) * i / n), r_in * math.sin(a1 - (a1 - a0) * i / n)) for i in range(n + 1)]
                solids.append(bd.Pos(0, 0, z0) * bd.extrude(bd.Polygon(*(outer + inner), align=None), amount=s))
            th = th1
    base = _tube(r_out, r_in, 0.01, z0 - 0.01)
    return base.fuse(*solids).clean()


# ------------------------------------------------------------------------------
# Drill head
# ------------------------------------------------------------------------------
def build_head(p: Dict[str, Any], d: Dict[str, Any], zc: float, rail_x: float) -> Dict[str, List[PartT]]:
    hd = p["drill_head"]
    H = d["head"]
    st = {k: zc + v for k, v in H["stack_mm"].items()}
    plan = H["plan_mm"]
    sp = d["hammer"]["spring"]
    wall = hd["housing_wall_m"] * 1e3
    carriage: List[PartT] = []
    spindle: List[PartT] = []
    hammer: List[PartT] = []
    cam: List[PartT] = []

    # -------- Spindle: DIN 5480 spline below, Ø20 journal above, integral spring seat, 24T gear on top
    r_maj = hd["spline_major_m"] * 1e3 / 2
    r_root = H["spline"]["root_d_m"] * 1e3 / 2
    r_ref = H["spline"]["pitch_d_m"] * 1e3 / 2
    z_sp = int(hd["spline_teeth"])
    r_bore = hd["spindle_bore_m"] * 1e3 / 2
    r_j = hd["spindle_journal_m"] * 1e3 / 2
    lift = p["actuators"]["percussive"]["cam_lift_m"] * 1e3
    z_spl_top = st["hammer_top"] + lift + 2.0
    shaft = _spline(z_sp, r_maj, r_root, r_ref, z_spl_top - st["web_top"], st["web_top"])
    shaft = shaft + _cyl(r_maj, st["spring_top"] - z_spl_top + 0.5, z_spl_top - 0.5)
    seat = _cyl(sp["coil_d_mm"] / 2 + sp["wire_mm"], st["seat_top"] - st["spring_top"], st["spring_top"])
    seat = bd.chamfer(seat.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.Z)[-1], length=1.0)
    shaft = shaft + seat + _cyl(r_j, st["gear_top"] - st["seat_top"], st["seat_top"])
    shaft = shaft - _cyl(r_bore, st["gear_top"] - st["web_top"] + 2, st["web_top"] - 1)
    shaft = shaft - bd.Pos(-r_j, 0, 0.5 * (st["slip_bottom"] + st["slip_top"])) * bd.Rot(0, 90, 0) * bd.Cylinder(1.5, r_j * 2)
    spindle.append(("Splined_Hollow_Spindle", shaft, "steel_shaft"))

    m_sp = H["rotary"]["spur_module_m"] * 1e3
    z_p, z_g = H["rotary"]["spur_teeth"]
    b_sp = st["gear_top"] - st["gear_bottom"]
    spindle.append((f"Spindle_Gear_{z_g}T", _gear(m_sp, z_g, b_sp, st["gear_bottom"], bore=r_j - 0.05), "brass_gear"))
    spindle.append(("Slip_Ring_Rotor", _tube(r_j + 3.0, r_j, st["slip_top"] - st["slip_bottom"] - 2, st["slip_bottom"] + 1), "copper_winding"))

    # -------- Anvil (slides on the spline; hex socket for the auger shank; WOB web; strike face)
    r_nose = 15.0
    r_anv = 22.0
    anvil = _cyl(r_nose, st["anvil_shoulder"] - st["anvil_bottom"], st["anvil_bottom"])
    anvil = anvil + _cyl(r_anv, st["anvil_top"] - st["anvil_shoulder"], st["anvil_shoulder"])
    anvil = bd.chamfer(anvil.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.Z)[-1], length=1.5)
    hex_sock = bd.Pos(0, 0, st["anvil_bottom"] - 1) * bd.extrude(bd.RegularPolygon(8.1, 6), st["hex_top"] - st["anvil_bottom"] + 1)
    anvil = anvil - hex_sock - _cyl(r_bore, 8, st["hex_top"] - 1)
    anvil = anvil - _spline(z_sp, r_maj + 0.1, r_root + 0.1, r_ref + 0.1, st["anvil_top"] - st["web_top"] + 1, st["web_top"])
    if "string" in d:  # cross hole for the auger retention pin (src/design_string.py)
        sh = d["string"]["shank"]
        anvil = anvil - bd.Pos(0, 0, zc + sh["pin_z_mm"]) * bd.Rot(0, 90, 0) * bd.Cylinder(sh["pin_d_m"] * 1e3 / 2 + 0.02, 2 * r_nose + 2)
    spindle.append(("Anvil", anvil, "flute_hardened"))

    # -------- Hammer (splined bore) + two stud-type follower rollers
    r_h = hd["hammer_od_m"] * 1e3 / 2
    hm = _cyl(r_h, st["hammer_top"] - st["anvil_top"], st["anvil_top"])
    hm = bd.chamfer(hm.edges().filter_by(bd.GeomType.CIRCLE), length=1.0)
    hm = hm - _spline(z_sp, r_maj + 0.1, r_root + 0.1, r_ref + 0.1, st["hammer_top"] - st["anvil_top"] + 2, st["anvil_top"] - 1)
    hammer.append(("Hammer_Mass", hm, "gearbox_gunmetal"))
    cm = H["cam"]
    r_m = cm["mean_radius_m"] * 1e3
    r_rl, w_rl = cm["roller_r_m"] * 1e3, cm["roller_w_m"] * 1e3
    home_deg = cm["rise_deg"] + 0.5 * cm["dwell_deg"]          # rollers rest in the dwell, 1 mm above the track
    for i in range(cm["lobes"]):
        ang = home_deg + 360.0 * i / cm["lobes"]
        stud = bd.Pos(r_h - 1 + 0.5 * (r_m - w_rl / 2 - r_h + 1), 0, st["roller_z"]) * bd.Rot(0, 90, 0) * bd.Cylinder(3.0, r_m - w_rl / 2 - r_h + 1)
        roller = bd.Pos(r_m, 0, st["roller_z"]) * bd.Rot(0, 90, 0) * bd.Cylinder(r_rl, w_rl)
        roller = bd.chamfer(roller.edges(), length=0.5)
        hammer.append((f"Cam_Follower_Stud_{i + 1}", bd.Rot(0, 0, ang) * stud, "steel_shaft"))
        hammer.append((f"Cam_Follower_Roller_{i + 1}", bd.Rot(0, 0, ang) * roller, "tungsten_carbide"))

    # -------- Hammer spring (squared & ground), turns with the spindle
    d_w = sp["wire_mm"]
    spindle.append(("Hammer_Compression_Spring",
                    _coil(sp["coil_d_mm"] / 2, d_w / 2, st["hammer_top"], st["spring_top"] - st["hammer_top"], sp["active_coils"] + 2),
                    "spring_amber"))

    # -------- Barrel cam: 61812 on a stationary spigot, 54T ring gear, two cycloidal lobes
    bc = hd["cam_bearing"]
    r_bi, r_bo, b_b = bc["d_m"] * 1e3 / 2, bc["D_m"] * 1e3 / 2, bc["B_m"] * 1e3
    z_b0 = st["track_base"] - 2.0 - b_b
    m_c = H["percussive"]["module_m"] * 1e3
    z_cg = H["percussive"]["cam_gear_teeth"]
    cam_gear = _gear(m_c, z_cg, st["track_base"] - st["cam_gear_bottom"], st["cam_gear_bottom"])
    cam_gear = cam_gear - _cyl(r_bo, st["track_base"] - st["cam_gear_bottom"] + 1, st["cam_gear_bottom"] - 0.5)
    cam_gear = cam_gear - _cyl(r_bo + 0.5, z_b0 - st["cam_gear_bottom"] + 0.5, st["cam_gear_bottom"] - 0.5)
    track_in, track_out = r_bi, r_m + w_rl / 2 + 1.0
    lip = _tube(r_bo + 0.5, track_in, 1.5, st["track_base"] - 1.5)
    track = _cam_track(cm["profile_deg_mm"], cm["lobes"], track_in, track_out, st["track_base"])
    cam.append(("Barrel_Cam_Ring_Gear", cam_gear + lip, "cam_crimson"))
    cam.append(("Barrel_Cam_Track_2_Lobe", track, "flute_hardened"))
    carriage += _bearing("Cam_Bearing_61812", 2 * r_bi, 2 * r_bo, b_b, z_b0)
    carriage.append(("Cam_Bearing_Spigot", _tube(r_bi, r_h + 0.8, z_b0 + b_b - st["housing_bottom"] - wall, st["housing_bottom"] + wall)
                     + _tube(r_bi + 2.5, r_h + 0.8, 2.0, z_b0 - 2.0), "titanium_frame"))

    # -------- Percussive drive: 18T pinion + secondary BLDC (pinion end down)
    py = plan["percussive_axis_y"]
    pc = H["percussive"]
    z_pp = pc["pinion_teeth"]
    carriage.append((f"Percussive_Pinion_{z_pp}T", _gear(m_c, z_pp, st["track_base"] - st["cam_gear_bottom"], st["cam_gear_bottom"], 0, py, rot=10.0),
                     "brass_gear"))
    l2 = pc["rotor_l_m"] * 1e3
    z_m2 = st["track_base"] + 4.0
    carriage += _motor("Percussive", 0, py, z_m2, pc["stator_od_m"] * 1e3, pc["rotor_d_m"] * 1e3, l2)

    # -------- Spindle bearings: 7004 ACD pair (DB) + floating 61804, slip ring stator
    sb, ub = hd["spindle_bearing"], hd["upper_bearing"]
    carriage += _bearing("Spindle_Bearing_7004ACD_A", sb["d_m"] * 1e3, sb["D_m"] * 1e3, sb["B_m"] * 1e3, st["pair_bottom"])
    carriage += _bearing("Spindle_Bearing_7004ACD_B", sb["d_m"] * 1e3, sb["D_m"] * 1e3, sb["B_m"] * 1e3, st["pair_bottom"] + sb["B_m"] * 1e3)
    carriage += _bearing("Upper_Bearing_61804", ub["d_m"] * 1e3, ub["D_m"] * 1e3, ub["B_m"] * 1e3, st["upper_bottom"])
    h_sr = st["slip_top"] - st["slip_bottom"]
    carriage.append(("Slip_Ring_Stator", _tube(r_j + 9.0, r_j + 3.5, h_sr, st["slip_bottom"]), "gold_kapton"))
    carriage.append(("Slip_Ring_Brush_Block", bd.Pos(-(r_j + 11.0), 0, st["slip_bottom"] + h_sr / 2) * bd.Box(4.0, 12.0, h_sr - 4), "dark_anodized"))

    # -------- Rotary drive: 12T pinion + 8:1 planetary (sun/3 planets/ring) + frameless BLDC
    ax = plan["rotary_axis_x"]
    rt = H["rotary"]
    carriage.append((f"Primary_Pinion_{z_p}T", _gear(m_sp, z_p, b_sp, st["gear_bottom"], ax, 0, bore=0, rot=15.0), "brass_gear"))
    m_pl = rt["planetary_module_m"] * 1e3
    zs, zpl, zr = rt["planetary_teeth"]
    b_pl = (st["planetary_top"] - st["gear_top"] - 12.0) / 2
    z_pl0 = st["gear_top"] + 6.0
    r_ring = m_pl * zr / 2 + 1.25 * m_pl + 2.0
    carriage.append(("Planetary_Ring_Gear_84T", _ring_gear(m_pl, zr, b_pl, z_pl0, r_ring, ax, 0), "gearbox_gunmetal"))
    carriage.append((f"Planetary_Sun_{zs}T", _gear(m_pl, zs, b_pl, z_pl0, ax, 0), "steel_shaft"))
    a_pl = m_pl * (zs + zpl) / 2
    for i in range(rt["planets"]):
        ang = 360.0 * i / rt["planets"]
        x = ax + a_pl * math.cos(math.radians(ang))
        y = a_pl * math.sin(math.radians(ang))
        rot = 180.0 / zpl if zpl % 2 == 0 else 0.0
        carriage.append((f"Planet_Gear_{zpl}T_{i + 1}", _gear(m_pl, zpl, b_pl, z_pl0, x, y, bore=1.5, rot=rot), "brass_gear"))
    carrier = _cyl(a_pl + 3.0, 3.0, z_pl0 - 4.0, ax, 0) + _cyl(3.0, z_pl0 - 4.0 - st["gear_bottom"], st["gear_bottom"], ax, 0)
    carriage.append(("Planet_Carrier_Output", carrier, "steel_shaft"))
    carriage.append(("Planetary_Cover", _tube(r_ring, m_pl * zs / 2 + 1.0, 3.0, z_pl0 + b_pl + 0.5, ax, 0), "gearbox_gunmetal"))
    carriage += _motor("Rotary", ax, 0, st["planetary_top"] + 2.0, rt["stator_od_m"] * 1e3, rt["rotor_d_m"] * 1e3, rt["rotor_l_m"] * 1e3)

    # -------- Housing (Ti shell): lower cam body + percussive pod + spindle column + rotary pod
    r_low = plan["housing_r_lower"]
    z_low_top = st["hammer_top"] + lift + 6.0
    r_pod2 = pc["stator_od_m"] * 1e3 / 2 + 1.5 + wall
    z_pod2_top = z_m2 + l2 + 18.0 + wall
    r_col = r_j + 14.0 + wall
    r_pod1 = max(rt["stator_od_m"] * 1e3 / 2, r_ring) + 1.5 + wall
    z_pod1_bot = st["gear_bottom"] - 3.0 - wall
    shells = [  # (r, x, y, z0, z1)
        (r_low, 0, 0, st["housing_bottom"], z_low_top),
        (r_pod2, 0, py, st["housing_bottom"], z_pod2_top),
        (r_col, 0, 0, z_low_top - 2, st["gear_top"] + 3.0 + wall),
        (r_pod1, ax, 0, z_pod1_bot, st["housing_top"]),
    ]
    outer = None
    inner = None
    for r, x, y, z0, z1 in shells:
        o = _cyl(r, z1 - z0, z0, x, y)
        try:
            o = bd.fillet(o.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.Z)[-1], radius=min(4.0, wall + 1))
        except Exception:
            pass
        i = _cyl(r - wall, z1 - z0 - 2 * wall, z0 + wall, x, y)
        outer = o if outer is None else outer + o
        inner = i if inner is None else inner + i
    housing = outer - inner
    housing = housing - _cyl(r_nose + 1.0, wall + 2, st["housing_bottom"] - 1)
    housing = housing - bd.Pos(-(r_col - wall / 2), 0, st["slip_bottom"] + h_sr / 2) * bd.Rot(0, 90, 0) * bd.Cylinder(4.0, wall + 4)
    carriage.append(("Head_Housing_Ti", housing, "titanium_frame"))
    carriage.append(("Slip_Ring_Cable_Gland", bd.Pos(-(r_col + 4.0), 0, st["slip_bottom"] + h_sr / 2) * bd.Rot(0, 90, 0) * bd.Cylinder(5.0, 10.0),
                     "dark_anodized"))
    carriage.append(("Spindle_Bearing_Bulkhead", _tube(r_col - wall, sb["D_m"] * 1e3 / 2, 2 * sb["B_m"] * 1e3 + 4, st["pair_bottom"] - 2), "titanium_frame"))

    # -------- Nose: labyrinth + PTFE lip seal cartridge, stationary bellows, dust cap riding on the anvil
    z_n0, z_n1 = st["housing_bottom"] - 2.0, st["nose_top"]
    nose = _tube(r_nose + 9.0, r_nose + 0.3, z_n1 - z_n0, z_n0)
    for k in range(3):
        nose = nose - _tube(r_nose + 2.0, r_nose, 1.2, z_n0 + 3.0 + 2.2 * k)
    carriage.append(("Labyrinth_Seal_Cartridge", nose, "dark_anodized"))
    carriage.append(("PTFE_Lip_Seal", _tube(r_nose + 2.5, r_nose + 0.05, 1.5, z_n1 - 2.5), "radiator_white"))
    z_cap = st["anvil_bottom"] + 6.0
    carriage.append(("Dust_Cap_PTFE_Lip", _tube(r_nose + 6.0, r_nose + 0.05, 3.0, z_cap), "radiator_white"))
    bel = None
    n_conv = 5
    pitch = (z_n0 - (z_cap + 3.0)) / n_conv
    for k in range(n_conv):
        zk = z_cap + 3.0 + k * pitch
        ring = bd.Pos(0, 0, zk + pitch / 2) * bd.Torus(r_nose + 6.0, pitch / 2 * 0.95)
        bel = ring if bel is None else bel + ring
    bel = bel - _cyl(r_nose + 4.0, z_n0 - z_cap, z_cap)
    carriage.append(("Dust_Bellows", bel, "tungsten_carbide"))
    carriage.append(("Diaphragm_Flexure", _tube(r_low - 6.0, r_nose + 9.0, 0.8, st["base_plate"]), "steel_bristle"))

    # -------- Carriage frame: base plate, top yoke, tie rods, LM16UU blocks, load cell, Tr16x4 nut
    ly = plan["leadscrew_y"]
    lb = hd["linear_bearing"]
    r_lb, l_lb = lb["D_m"] * 1e3 / 2, lb["L_m"] * 1e3
    y0, y1 = -60.0, ly + 14.0
    slot = p["drill_mast"]["rail_bracket_neck_m"] * 1e3 + 2.0          # open LM16UU-OP: brackets pass with 1 mm each side
    def plate(z0, h, hole_r):
        pl = bd.Pos(0, 0.5 * (y0 + y1), z0 + h / 2) * bd.Box(220.0, y1 - y0, h)
        pl = bd.fillet(pl.edges().filter_by(bd.Axis.Z), radius=8.0)
        for sx in (-1, 1):
            pl = pl - _cyl(8.5, h + 2, z0 - 1, sx * rail_x)
            pl = pl - bd.Pos(sx * rail_x, 0.5 * (y1 + 1.0), z0 + h / 2) * bd.Box(slot, y1 + 1.0, h + 2)   # rail-bracket neck slot
        pl = pl - _cyl(9.0, h + 2, z0 - 1, 0, ly)
        if hole_r > 0:
            pl = pl - _cyl(hole_r, h + 2, z0 - 1)
        return pl
    carriage.append(("Carriage_Base_Plate", plate(zc, st["base_plate"] - zc, r_low - 6.0), "titanium_frame"))
    z_yoke0 = st["load_cell_top"]
    carriage.append(("Carriage_Top_Yoke", plate(z_yoke0, st["yoke_top"] - z_yoke0, 0.0), "titanium_frame"))
    for ix, sx in enumerate((-100.0, 100.0)):
        for iy, sy in enumerate((-48.0, ly)):
            carriage.append((f"Carriage_Tie_Rod_{ix}{iy}", _cyl(5.0, z_yoke0 - st["base_plate"], st["base_plate"], sx, sy), "dark_anodized"))
    for side, sx in (("L", -rail_x), ("R", rail_x)):
        for tag, z0 in (("Lower", st["base_plate"]), ("Upper", z_yoke0 - l_lb)):
            blk = bd.Pos(sx, 0, z0 + l_lb / 2) * bd.Box(2 * r_lb + 8, 2 * r_lb + 8, l_lb)
            cut = bd.Pos(sx, r_lb + 3, z0 + l_lb / 2) * bd.Box(slot, 2 * r_lb + 6, l_lb + 4)
            blk = bd.fillet(blk.edges().filter_by(bd.Axis.Z), radius=3.0) - _cyl(r_lb, l_lb + 2, z0 - 1, sx) - cut
            carriage.append((f"Linear_Bearing_Block_{side}_{tag}", blk, "dark_anodized"))
            carriage.append((f"LM16UU_OP_{side}_{tag}", _tube(r_lb, 8.05, l_lb - 2, z0 + 1, sx) - cut, "gearbox_gunmetal"))
        carriage.append((f"Rail_Wiper_{side}", _tube(r_lb, 8.0, 2.0, zc - 2.0, sx) - bd.Pos(sx, r_lb, zc - 1.0) * bd.Box(slot, 2 * r_lb, 4), "radiator_white"))
    l_nut = H["nut"]["length_m"] * 1e3
    nut = _tube(14.0, 6.9, l_nut, st["yoke_top"] - l_nut + 6.0, 0, ly) + _tube(18.0, 6.9, 5.0, st["yoke_top"], 0, ly)
    carriage.append(("Leadscrew_Nut_Tr16x4_Bronze", nut, "chute_bronze"))
    lc = _cyl(25.0, st["load_cell_top"] - st["housing_top"] - 4, st["housing_top"] + 2) + _cyl(8.0, 2.2, st["housing_top"]) \
        + _cyl(8.0, 2.2, st["load_cell_top"] - 2.2)
    carriage.append((f"WOB_Load_Cell_{int(H['load_cell_n'])}N", lc, "gold_kapton"))
    return {"carriage": carriage, "spindle": spindle, "hammer": hammer, "cam": cam}


# Small repetitive features (ball sets, bellows convolutions, spring wire) tessellate coarser for the web
COARSE_MESH = ("_Balls", "Dust_Bellows", "Hammer_Compression_Spring")


SECTION_CUT = ("Head_Housing_Ti", "Labyrinth_Seal_Cartridge", "Dust_Bellows", "Carriage_Base_Plate", "Carriage_Top_Yoke",
               "Cam_Bearing_Spigot", "Spindle_Bearing_Bulkhead", "Diaphragm_Flexure", "Slip_Ring_Stator",
               "Planetary_Ring_Gear_84T", "Rotary_BLDC_Stator", "Percussive_BLDC_Stator", "Barrel_Cam_Ring_Gear",
               "Anvil", "Hammer_Mass", "Splined_Hollow_Spindle", "PTFE_Lip_Seal", "Dust_Cap_PTFE_Lip")


def section(parts: List[PartT], zc: float) -> List[PartT]:
    """Quarter cut-away (x > 0, y < 0 removed) of the shells and the coaxial stack, so the internals show."""
    q = bd.Pos(150, -150, zc + 150) * bd.Box(300, 300, 600)
    out = []
    for name, shape, mat in parts:
        if name in SECTION_CUT:
            try:
                shape = shape - q
            except Exception:
                pass
        out.append((name, shape, mat))
    return out
