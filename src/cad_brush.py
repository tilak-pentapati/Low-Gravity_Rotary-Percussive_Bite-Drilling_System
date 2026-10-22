"""
Passive brushing station - detailed build123d geometry (mm, Z-up).

All sizes come from src/design_brush.py (tuft count, pitch / tip / hub radii, tuft lean, bristle length,
face width, arm, torsion spring, bushings, chute, cup, stow angle) and config/system_params.yaml.

Rotating group (Brush_Gear_Node, local frame: axle along X through the origin):
  Brush_Axle, Brush_Hub, Brush_Hub_End_Cap_L/R, Bristle_Tuft_01..N (worm-wheel teeth: tufts lean at the
  flight lead angle and sit on the derived pitch radius), Brush_Axle_Spacer_L/R
Frame group (world coordinates, brush axle at (0, -a, z_brush)):
  Brush_Arm_L/R            pivot arms hanging below the axle (line-of-centres force acts at 90 deg to them)
  Brush_Axle_Bushing_L/R   Vespel SP-3 flanged bushings
  Brush_Pivot_Pin_L/R, Brush_Pivot_Bracket_L/R, Brush_Torsion_Spring (preload = required bristle force)
  Stow_Launch_Lock         holds the arm in the stowed / dry-heat bake-out position until first use
  Sample_Transfer_Chute    phosphor-bronze U-trough at the derived slope, inlet notch around the auger
  Chute_Post_1/2, Sample_Port_Funnel (bronze collar over the foot-plate port, under the chute outlet)
  Carousel_Index_Motor (on the foot plate at the carousel axis), Carousel_Hub_Bearing (under the plate)
Carousel group (Carousel_Node, local frame: carousel axis through the origin, world heights):
  Carousel_Turntable, Carousel_Hub, Sample_Cup_01..N, Cup_Seal_Ring_01..N. Station 0 (blank) sits under the
  port at 90 deg (towards the auger); cup s sits at 90 deg - s x 360 / N_st, so turning the group by +s x pitch
  about +Z brings cup s under the port. The foot plate is the cups' cover (src/cad_mast.py cuts the port).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import build123d as bd

PartT = tuple
ALIGN_BASE = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)


def _cyl(r, h, z0, x=0.0, y=0.0):
    return bd.Pos(x, y, z0) * bd.Cylinder(r, h, align=ALIGN_BASE)


def _xcyl(r, length, cx=0.0, cy=0.0, cz=0.0):
    return bd.Pos(cx, cy, cz) * bd.Rot(0, 90, 0) * bd.Cylinder(r, length)


def _xtube(ro, ri, length, cx=0.0, cy=0.0, cz=0.0):
    return _xcyl(ro, length, cx, cy, cz) - _xcyl(ri, length + 0.2, cx, cy, cz)


def geometry(p: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, float]:
    """Shared brush-station dimensions in mm (also used for the layout / manifest)."""
    br = d["brush"]
    m, b, arm, cup = br["mesh"], br["bristle"], br["arm"], br["cup"]
    g = {
        "a": m["axle_offset_m"] * 1e3, "r_p": m["r_p_m"] * 1e3, "r_t": m["r_tip_m"] * 1e3, "r_h": m["r_hub_m"] * 1e3,
        "n": m["tufts"], "lean": m["lean_deg"], "s_t": m["tuft_thickness_m"] * 1e3, "face": b["face_m"] * 1e3,
        "l_arm": arm["length_m"] * 1e3, "pin": arm["pivot_pin_m"] * 1e3, "ax_d": arm["axle_d_m"] * 1e3,
        "bush_l": arm["bushing_len_m"] * 1e3, "cup_r": cup["id_m"] * 1e3 / 2, "z": p["cad_layout"]["brush_axle_height_m"] * 1e3,
        "arm_t": 4.0, "br_t": 5.0,
    }
    sp = arm["spring"]
    g["spring_len"] = (sp["active_coils"] + 1) * sp["wire_m"] * 1e3
    g["x_arm"] = g["cup_r"] + 1.0 + p["actuators"]["brush_station"]["chute_side_clearance_m"] * 1e3 + g["arm_t"] / 2
    g["x_br"] = g["x_arm"] + g["arm_t"] / 2 + g["spring_len"] + 0.5 + g["br_t"] / 2
    return g


def build_brush_local(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Rotating brush (worm wheel) in its local frame: axle along X through the origin."""
    g = geometry(p, d)
    r_h, r_t, face, s_t = g["r_h"], g["r_t"], g["face"], g["s_t"]
    parts: List[PartT] = []
    half_ax = g["x_arm"] + g["arm_t"] / 2 + 1.5
    parts.append(("Brush_Axle", _xcyl(g["ax_d"] / 2, 2 * half_ax), "steel_shaft"))
    hub = _xcyl(r_h, face + 2.0) - _xcyl(g["ax_d"] / 2 + 0.02, face + 4.0)
    anchor = 3.0
    tufts = []
    for k in range(g["n"]):
        blk = bd.Pos(0, 0, r_h - anchor + 0.5 * (r_t - r_h + anchor)) * bd.Box(face, s_t, r_t - r_h + anchor)
        for i in range(1, 8):                                   # bristle rows: thin slots across the tuft
            xs = -face / 2 + i * face / 8
            blk = blk - bd.Pos(xs, 0, r_h + 0.5 * (r_t - r_h) + 1.0) * bd.Box(0.5, s_t + 1, r_t - r_h)
        blk = bd.Rot(-90.0 + 360.0 * k / g["n"], 0, 0) * bd.Rot(0, 0, g["lean"]) * blk
        tufts.append(blk)
        hub = hub - bd.Rot(-90.0 + 360.0 * k / g["n"], 0, 0) * bd.Rot(0, 0, g["lean"]) * bd.Pos(0, 0, r_h - anchor / 2) * bd.Box(face + 0.1, s_t + 0.1, anchor + 0.1)
    parts.append(("Brush_Hub", hub, "brass_gear"))
    for i, t in enumerate(tufts):
        parts.append((f"Bristle_Tuft_{i + 1:02d}", t, "steel_bristle"))
    for side, sx in (("L", -1), ("R", 1)):
        cap = _xcyl(r_h + 1.0, 1.5, sx * (face / 2 + 1.75)) - _xcyl(g["ax_d"] / 2 + 0.02, 3.0, sx * (face / 2 + 1.75))
        parts.append((f"Brush_Hub_End_Cap_{side}", cap, "brass_gear"))
        x0, x1 = face / 2 + 2.5, g["x_arm"] - g["arm_t"] / 2 - 1.0
        parts.append((f"Brush_Axle_Spacer_{side}", _xtube(g["ax_d"] / 2 + 1.5, g["ax_d"] / 2 + 0.02, x1 - x0, sx * 0.5 * (x0 + x1)), "dark_anodized"))
    return parts


def _arm(g: Dict[str, float], x: float, by: float) -> bd.Part:
    l, w = g["l_arm"], 12.0
    slot = bd.extrude(bd.SlotCenterToCenter(l, w), amount=g["arm_t"] / 2, both=True)
    arm = bd.Pos(x, by, g["z"] - l / 2) * bd.Rot(0, 90, 0) * slot
    arm = arm - _xcyl(g["ax_d"] / 2 + 1.5, 10, x, by, g["z"]) - _xcyl(g["pin"] / 2 + 0.05, 10, x, by, g["z"] - l)
    return arm


def build_brush_frame(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Static (frame-mounted) brush-station parts in world coordinates."""
    g = geometry(p, d)
    br = d["brush"]
    ch, cup = br["chute"], br["cup"]
    dt = p["cad_layout"]["foot_top_m"] * 1e3
    by, bz = -g["a"], g["z"]
    z_piv = bz - g["l_arm"]
    parts: List[PartT] = []

    for side, sx in (("L", -1), ("R", 1)):
        xa, xb = sx * g["x_arm"], sx * g["x_br"]
        parts.append((f"Brush_Arm_{side}", _arm(g, xa, by), "titanium_frame"))
        bush = _xtube(g["ax_d"] / 2 + 1.5, g["ax_d"] / 2 + 0.02, g["bush_l"], xa, by, bz)
        bush = bush + _xtube(g["ax_d"] / 2 + 3.5, g["ax_d"] / 2 + 0.02, 1.0, xa - sx * (g["arm_t"] / 2 + 0.5), by, bz)
        parts.append((f"Brush_Axle_Bushing_{side}", bush, "vespel_brown"))
        x_in, x_out = xa - sx * (g["arm_t"] / 2 + 0.5), xb + sx * (g["br_t"] / 2 + 1.0)
        parts.append((f"Brush_Pivot_Pin_{side}", _xcyl(g["pin"] / 2, abs(x_out - x_in), 0.5 * (x_in + x_out), by, z_piv), "steel_shaft"))
        h = z_piv + 9.0 - dt
        post = bd.Pos(xb, by, dt + h / 2) * bd.Box(g["br_t"], 16.0, h)
        post = bd.fillet(post.edges().filter_by(bd.Axis.X).group_by(bd.Axis.Z)[-1], radius=7.5)
        post = post + bd.Pos(xb + sx * 4.0, by, dt + 2.0) * bd.Box(g["br_t"] + 8.0, 24.0, 4.0)
        parts.append((f"Brush_Pivot_Bracket_{side}", post - _xcyl(g["pin"] / 2 + 0.05, 10, xb, by, z_piv), "titanium_frame"))

    sp = br["arm"]["spring"]
    helix = bd.Helix(pitch=g["spring_len"] / (sp["active_coils"] + 0.5), height=g["spring_len"] - sp["wire_m"] * 1e3,
                     radius=sp["coil_d_m"] * 1e3 / 2)
    plane = bd.Plane(origin=helix @ 0, x_dir=(1, 0, 0), z_dir=helix % 0)
    with bd.BuildPart() as bp:
        with bd.BuildLine():
            bd.add(helix)
        with bd.BuildSketch(plane):
            bd.Circle(sp["wire_m"] * 1e3 / 2)
        bd.sweep(is_frenet=True)
    x_s0 = -(g["x_arm"] + g["arm_t"] / 2 + 0.25)
    spring = bd.Pos(x_s0, by, z_piv) * bd.Rot(0, -90, 0) * bp.part
    leg = bd.Pos(x_s0 - 0.5, by - 0.5 * (g["l_arm"] * 0.25), z_piv + sp["coil_d_m"] * 1e3 / 2) * bd.Box(sp["wire_m"] * 1e3, g["l_arm"] * 0.25, sp["wire_m"] * 1e3)
    parts.append(("Brush_Torsion_Spring", spring + leg, "spring_amber"))
    lock = _xcyl(4.0, 14.0, g["x_br"] + g["br_t"] / 2 + 7.0, by + 12.0, z_piv) + _xcyl(1.2, 8.0, g["x_br"] - 1.0, by + 12.0, z_piv)
    parts.append(("Stow_Launch_Lock", lock, "cam_crimson"))

    # Chute: U-trough at the derived slope, inlet notch around the auger
    th = ch["angle_deg"]
    w, wall, hw = ch["width_m"] * 1e3, ch["wall_m"] * 1e3, ch["wall_h_m"] * 1e3
    y_in, y_out = ch["y_in_m"] * 1e3, ch["y_out_m"] * 1e3
    y_start, y_end = 2.0, y_out + 0.25 * cup["id_m"] * 1e3
    length = (y_end - y_start) / math.cos(math.radians(th))
    trough = bd.Pos(0, -length / 2, (hw + wall) / 2) * bd.Box(w + 2 * wall, length, hw + wall)
    trough = trough - bd.Pos(0, -length / 2, wall + hw / 2 + 0.5) * bd.Box(w, length + 2, hw + 1)
    z0 = ch["z_in_m"] * 1e3 + (y_in - y_start) * math.tan(math.radians(th))
    trough = bd.Pos(0, -y_start, z0) * bd.Rot(th, 0, 0) * trough
    trough = trough - _cyl(y_in, 80.0, z0 - 60.0)
    parts.append(("Sample_Transfer_Chute", trough, "chute_bronze"))
    for i, sx in enumerate((-1, 1)):
        yp = y_in + 4.0
        zp = ch["z_in_m"] * 1e3 - (yp - y_in) * math.tan(math.radians(th))
        parts.append((f"Chute_Post_{i + 1}", _cyl(1.5, zp - dt, dt, sx * (w / 2 + wall + 1.5), -yp), "dark_anodized"))

    # Funnel collar over the foot-plate port: catches cuttings leaving the chute outlet
    r_c = cup["id_m"] * 1e3 / 2
    yc = -cup["y_m"] * 1e3
    z_fun = ch["z_out_m"] * 1e3 - (r_c + wall) * math.tan(math.radians(th)) - wall - 1.0 - dt   # under the trough's far edge
    parts.append(("Sample_Port_Funnel", _cyl(r_c + wall, z_fun, dt, 0, yc) - _cyl(r_c, z_fun + 2, dt - 1, 0, yc), "chute_bronze"))

    # Carousel drive: index motor on the plate at the carousel axis, hub bearing housing under the plate
    car = br["carousel"]
    ya = -car["axis_offset_m"] * 1e3
    mo = car["motor"]
    t_ft = p["drill_mast"]["foot_plate_thickness_m"] * 1e3
    motor = _cyl(mo["od_m"] * 1e3 / 2, mo["length_m"] * 1e3, dt, 0, ya)
    motor = bd.fillet(motor.edges().group_by(bd.Axis.Z)[-1], radius=2.0)
    parts.append(("Carousel_Index_Motor", motor, "bldc_emerald"))
    z_hub0 = car["turntable_z0_m"] * 1e3 + car["turntable_t_m"] * 1e3
    h_bear = 8.0
    parts.append(("Carousel_Hub_Bearing", _cyl(14.0, h_bear, dt - t_ft - h_bear, 0, ya) - _cyl(8.0, h_bear + 2, dt - t_ft - h_bear - 1, 0, ya),
                  "gearbox_gunmetal"))
    return parts


def build_carousel_local(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Rotating carousel (turntable, hub, cups, seal rings) about its own axis at the local origin (world heights)."""
    br = d["brush"]
    cup, car = br["cup"], br["carousel"]
    wall = cup["wall_m"] * 1e3
    r_c = cup["id_m"] * 1e3 / 2
    r_f = car["flange_d_m"] * 1e3 / 2
    z_rim = cup["rim_m"] * 1e3
    z_c0 = cup["z0_m"] * 1e3
    z_tt0 = car["turntable_z0_m"] * 1e3
    t_tt = car["turntable_t_m"] * 1e3
    r_pc = car["pitch_radius_m"] * 1e3
    dt = p["cad_layout"]["foot_top_m"] * 1e3
    t_ft = p["drill_mast"]["foot_plate_thickness_m"] * 1e3
    parts: List[PartT] = []
    tt = _cyl(car["turntable_r_m"] * 1e3, t_tt, z_tt0)
    for k in range(6):                                          # lightening holes between the hub and the cup ring
        ang = math.radians(30 + 60 * k)
        rr = 0.5 * (12.0 + r_pc - r_f)
        tt = tt - _cyl(0.3 * (r_pc - r_f - 12.0), t_tt + 2, z_tt0 - 1, rr * math.cos(ang), rr * math.sin(ang))
    parts.append(("Carousel_Turntable", tt, "dark_anodized"))
    z_top = dt - t_ft - 8.0                                      # into the hub bearing under the plate
    parts.append(("Carousel_Hub", _cyl(7.9, z_top - (z_tt0 + t_tt), z_tt0 + t_tt) + _cyl(12.0, 4.0, z_tt0 + t_tt), "steel_shaft"))
    th = 2 * math.pi / car["stations"]
    for s_ in range(1, car["cups"] + 1):
        a = math.pi / 2 - s_ * th
        x, y = r_pc * math.cos(a), r_pc * math.sin(a)
        c = _cyl(r_c + wall, z_rim - z_c0, z_c0, x, y) - _cyl(r_c, z_rim - z_c0, z_c0 + wall, x, y)
        c = c + (_cyl(r_f, 1.0, z_rim - 1.0, x, y) - _cyl(r_c, 2.0, z_rim - 1.5, x, y))
        parts.append((f"Sample_Cup_{s_:02d}", c, "gold_kapton"))
        seal = _cyl(r_f - 0.4, car["seal_gap_m"] * 1e3, z_rim, x, y) - _cyl(r_c + 0.4, 2.0, z_rim - 0.5, x, y)
        parts.append((f"Cup_Seal_Ring_{s_:02d}", seal, "ptfe_white"))
    return parts


EXPLODE: Dict[str, Tuple[float, float, float]] = {  # (dx, dy, dz) mm for the exploded view, by name prefix
    "Brush_Arm_L": (-25, 0, 0), "Brush_Arm_R": (25, 0, 0),
    "Brush_Axle_Bushing_L": (-15, 0, 0), "Brush_Axle_Bushing_R": (15, 0, 0),
    "Brush_Pivot_Pin_L": (-45, 0, 0), "Brush_Pivot_Pin_R": (45, 0, 0),
    "Brush_Pivot_Bracket_L": (-35, 0, 0), "Brush_Pivot_Bracket_R": (35, 0, 0),
    "Brush_Torsion_Spring": (-30, 0, -15), "Stow_Launch_Lock": (55, 0, 0),
    "Brush_Axle": (75, 0, 0), "Brush_Axle_Spacer_L": (-22, 0, 0), "Brush_Axle_Spacer_R": (22, 0, 0),
    "Brush_Hub_End_Cap_L": (-14, 0, 0), "Brush_Hub_End_Cap_R": (14, 0, 0),   # hub + tufts stay in mesh with the flights
    "Sample_Transfer_Chute": (0, -10, -30), "Chute_Post": (0, -10, -45),
    "Sample_Port_Funnel": (0, -10, -15), "Carousel_Index_Motor": (0, 0, 45), "Carousel_Hub_Bearing": (0, 0, -20),
    "Carousel_Turntable": (0, 0, -60), "Carousel_Hub": (0, 0, -40), "Sample_Cup_": (0, 0, -40), "Cup_Seal_Ring_": (0, 0, -30),
}


def explode_offset(name: str) -> Tuple[float, float, float]:
    for k in sorted(EXPLODE, key=len, reverse=True):
        if name.startswith(k):
            return EXPLODE[k]
    return (0.0, 0.0, 0.0)
