"""
Deployable drill mast and lander-deck interface - build123d geometry (mm, Z-up).

All sizes come from src/design_mast.py (spine section, rail-bracket count, latch height, hinge and latch pins,
hold-down stations, actuator size, deck length, leg footprint) and config/system_params.yaml.

Mast group (rotates about the hinge; drawn deployed):
  Mast_Spine_<b>x<h>x<t>      rectangular tube, extrusion corner radii
  Rail_Bracket_1..n            Al brackets whose necks pass the open-bushing slot, holding the Ø16 rails
  Guide_Rail_L/R, Leadscrew_Tr16x4, Leadscrew_Fixed_Bearing (duplex, top), Leadscrew_Support_Bearing (foot)
  Mast_Crosshead, Z_Feed_Motor, Mast_Foot_Plate (guide-bushing boss and brushing station sit on it)
  Hinge_Fork_L/R, Latch_Lug_L/R, Hold_Down_Cone_1..n
Lander group (static):
  Lander_Deck_Panel, Hinge_Tongue_L/R, Hinge_Pin_L/R, Deploy_Actuator (BLDC + strain-wave gear on the hinge axis),
  Latch_Post_L/R + braces, Latch_Pin_L/R, Hold_Down_Pedestal_n + SMA_Frangibolt_n, Leg_n_Main/Brace/Footpad
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import build123d as bd

PartT = tuple
ALIGN_BASE = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)


def _cyl(r, h, z0, x=0.0, y=0.0):
    return bd.Pos(x, y, z0) * bd.Cylinder(r, h, align=ALIGN_BASE)


def _xcyl(r, length, cx=0.0, cy=0.0, cz=0.0):
    return bd.Pos(cx, cy, cz) * bd.Rot(0, 90, 0) * bd.Cylinder(r, length)


def _ycyl(r, length, cx=0.0, cy=0.0, cz=0.0):
    return bd.Pos(cx, cy, cz) * bd.Rot(90, 0, 0) * bd.Cylinder(r, length)


def _box(x0, x1, y0, y1, z0, z1, r: float = 0.0):
    """Axis-aligned box from corner ranges; optional fillet on the vertical edges."""
    b = bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * bd.Box(x1 - x0, y1 - y0, z1 - z0)
    if r > 0:
        b = bd.fillet(b.edges().filter_by(bd.Axis.Z), radius=min(r, 0.49 * min(x1 - x0, y1 - y0)))
    return b


def _rod(p1: Sequence[float], p2: Sequence[float], r: float) -> bd.Solid:
    a, b = bd.Vector(*p1), bd.Vector(*p2)
    return bd.Solid.make_cylinder(r, (b - a).length, bd.Plane(origin=a, z_dir=b - a))


def _lug(x0, x1, yc, zc, r_end, y_root, z_root0, z_root1, hole_r):
    """Clevis lug in the YZ plane (thickness along x): root block on the spine, rounded end around the pin."""
    root = _box(x0, x1, min(y_root, yc), max(y_root, yc), z_root0, z_root1)
    end = bd.Pos((x0 + x1) / 2, yc, zc) * bd.Rot(0, 90, 0) * bd.Cylinder(r_end, x1 - x0)
    web = _box(x0, x1, min(y_root, yc), max(y_root, yc), zc - r_end, zc + r_end)
    return (root + web + end) - _xcyl(hole_r, x1 - x0 + 2, (x0 + x1) / 2, yc, zc)


def geometry(d: Dict[str, Any]) -> Dict[str, Any]:
    """Mast geometry in mm (lists stay lists)."""
    g = {k: (v * 1e3 if isinstance(v, float) else v) for k, v in d["mast"]["geometry_m"].items()}
    g["hold_downs_z"] = [z * 1e3 for z in d["mast"]["geometry_m"]["hold_downs_z"]]
    g["y_stow"] = [y * 1e3 for y in d["mast"]["geometry_m"]["y_stow"]]
    g["y_sc"] = g["y_spine_fore"] + g["h"] / 2
    return g


def build_mast(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Rotating mast group in the deployed pose (world mm)."""
    g = geometry(d)
    hd = p["drill_head"]
    r_lb = hd["linear_bearing"]["D_m"] * 1e3 / 2
    b, h, t = g["b"], g["h"], g["t"]
    ysf, yh, ysc = g["y_spine_fore"], g["y_hinge"], g["y_sc"]
    z0, ztop, zft, zxh = g["z_foot_bottom"], g["z_top"], g["z_foot_top"], g["z_crosshead_bottom"]
    rx, yls, hw = g["rail_x"], g["y_ls"], g["half_width"]
    parts: List[PartT] = []

    # Spine: rectangular hollow section with extrusion corner radii (outer 1.5 t, inner 0.5 t)
    outer = _box(-b / 2, b / 2, ysf, yh, z0, ztop, r=1.5 * t)
    inner = _box(-b / 2 + t, b / 2 - t, ysf + t, yh - t, z0 - 1, ztop + 1, r=0.5 * t)
    parts.append((f"Mast_Spine_{b:.0f}x{h:.0f}x{t:.0f}", outer - inner, "titanium_frame"))

    # Rail-support brackets: body on the spine fore face, necks through the open-bushing slot to the rails
    n = g["n_brackets"]
    neck, tb, ybr = g["bracket_neck"], g["bracket_t"], g["y_bracket"]
    for k in range(n):
        zk = zft + (k + 1) * (zxh - zft) / (n + 1)
        body = _box(-rx - neck / 2 - 4, rx + neck / 2 + 4, ybr, ysf, zk - tb / 2, zk + tb / 2, r=3.0)
        for sx in (-1, 1):
            body = body + _box(sx * rx - neck / 2, sx * rx + neck / 2, 0.0, ybr, zk - tb / 2, zk + tb / 2)
            body = body - _cyl(8.05, tb + 2, zk - tb / 2 - 1, sx * rx)
        parts.append((f"Rail_Bracket_{k + 1}", body, "dark_anodized"))
    for side, sx in (("L", -rx), ("R", rx)):
        parts.append((f"Guide_Rail_{side}", _cyl(8.0, zxh - zft, zft, sx), "steel_shaft"))
        blk = _box(sx - 18, sx + 18, -18, 18, zft, zft + 16, r=3.0) - _cyl(8.05, 18, zft - 1, sx)
        parts.append((f"Rail_Foot_Block_{side}", blk, "dark_anodized"))
    lead = p["actuators"]["z_axis_feed"]["lead_mm"]
    parts.append((f"Leadscrew_Tr16x{lead:.0f}", _cyl(8.0, zxh - zft - 12.0, zft + 12.0, 0, yls), "brass_gear"))
    sb = _box(-16, 16, yls - 16, yls + 16, zft, zft + 12, r=3.0) - _cyl(8.1, 14, zft - 1, 0, yls)
    parts.append(("Leadscrew_Support_Bearing", sb, "dark_anodized"))
    parts.append(("Leadscrew_Fixed_Bearing", _cyl(15.0, 16.0, zxh - 16.0, 0, yls) - _cyl(8.1, 18, zxh - 17, 0, yls), "gearbox_gunmetal"))

    # Crosshead and feed motor
    xh = _box(-hw, hw, -(r_lb + 4.0), yh, zxh, ztop, r=8.0)
    for sx in (-1, 1):
        xh = xh - _cyl(8.05, 4.0, zxh - 1, sx * rx)
    parts.append(("Mast_Crosshead", xh, "titanium_frame"))
    rm, lm = g["motor_d"] / 2, g["motor_l"]
    motor = _cyl(rm, lm, ztop, 0, yls)
    motor = bd.fillet(motor.edges().group_by(bd.Axis.Z)[-1], radius=3.0)
    parts.append(("Z_Feed_Motor", motor, "bldc_emerald"))
    parts.append(("Z_Feed_Motor_Band", _cyl(rm + 0.6, 12.0, ztop + 0.55 * lm, 0, yls) - _cyl(rm - 1.0, 14.0, ztop + 0.55 * lm - 1, 0, yls),
                  "copper_winding"))

    # Foot plate (guide-bushing boss and brushing station sit on it)
    r_boss = d["string"]["bushing"]["bore_m"] * 1e3 / 2 + 4.5
    foot = _box(-hw, hw, g["y_foot_fore"], yh, z0, zft, r=8.0) - _cyl(r_boss, g["foot_t"] + 2, z0 - 1)
    # Disc over the sample carousel: the plate is the cups' sealing cover, with one port under the chute outlet
    # and a bore for the index-motor shaft (src/design_brush.py, src/cad_brush.py)
    cup = d["brush"]["cup"]
    foot = foot + _cyl(g["foot_disc_r"], g["foot_t"], z0, 0, g["y_foot_fore"])
    foot = foot - _cyl(cup["id_m"] * 1e3 / 2, g["foot_t"] + 2, z0 - 1, 0, -cup["y_m"] * 1e3)
    foot = foot - _cyl(8.5, g["foot_t"] + 2, z0 - 1, 0, g["y_foot_fore"])
    parts.append(("Mast_Foot_Plate", foot, "titanium_frame"))

    # Hinge forks on the spine sides (double-shear clevis around the deck tongue)
    lt, dp = g["lug_t"], g["hinge_pin_d"]
    zh = g["z_hinge"]
    r_end = 1.25 * dp + 2.0
    for side, sx in (("L", -1), ("R", 1)):
        x_in = b / 2
        fork = None
        for k in (0, 2):
            xa, xb_ = x_in + k * lt + (0.5 if k else 0.0), x_in + (k + 1) * lt + (0.5 if k else 0.0)
            xa, xb_ = (xa, xb_) if sx > 0 else (-xb_, -xa)
            cheek = _lug(xa, xb_, yh, zh, r_end, ysc, zh, zh + 3 * r_end, dp / 2)
            fork = cheek if fork is None else fork + cheek
        xs = (x_in, x_in + 3 * lt + 0.5) if sx > 0 else (-(x_in + 3 * lt + 0.5), -x_in)
        fork = fork + _box(xs[0], xs[1], ysc, yh, zh + 2 * r_end, zh + 3 * r_end)
        parts.append((f"Hinge_Fork_{side}", fork, "titanium_frame"))
    # Latch lugs reaching to the posts
    zl, xp = g["z_latch"], g["x_post"]
    for side, sx in (("L", -1), ("R", 1)):
        xa, xb_ = sorted((sx * b / 2, sx * (xp - 21.0)))
        lug = _box(xa, xb_, ysc - 15, ysc + 15, zl - 12, zl + 12, r=0.0) - _xcyl(g["latch_pin_d"] / 2 + 0.1, abs(xb_ - xa) + 2, (xa + xb_) / 2, ysc, zl)
        parts.append((f"Latch_Lug_{side}", lug, "titanium_frame"))
    for k, zhd in enumerate(g["hold_downs_z"]):
        cone = bd.Pos(0, yh, zhd) * bd.Rot(-90, 0, 0) * bd.Cone(14.0, 7.0, g["stow_gap"] - 2.0, align=ALIGN_BASE)
        parts.append((f"Hold_Down_Cone_{k + 1}", cone, "gearbox_gunmetal"))
    return parts


def build_lander(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Static lander-deck interface: deck, hinge tongues + pins, actuator, latch posts, hold-downs, legs."""
    g = geometry(d)
    b, yh, ysc, zh, zd = g["b"], g["y_hinge"], g["y_sc"], g["z_hinge"], g["z_deck"]
    W, L, td = g["deck_width"], g["deck_length"], g["deck_t"]
    lt, dp = g["lug_t"], g["hinge_pin_d"]
    parts: List[PartT] = []
    zl, xp = g["z_latch"], g["x_post"]
    y0 = yh + 5.0                                            # deck fore edge just aft of the spine; horns carry the posts
    deck = _box(-W / 2, W / 2, y0, yh + L, zd - td, zd, r=12.0)
    for sx in (-1, 1):
        xa, xb_ = sorted((sx * (xp - 20.0), sx * W / 2))
        deck = deck + _box(xa, xb_, ysc - 25.0, y0 + 20.0, zd - td, zd)
    parts.append(("Lander_Deck_Panel", deck, "gold_kapton"))
    parts.append(("Deck_Face_Sheet", _box(-W / 2 + 2, W / 2 - 2, y0 + 2, yh + L - 2, zd - 0.5, zd + 0.3, r=10.0), "titanium_frame"))

    # Hinge: deck tongues inside the mast forks, pins, actuator on the +x hinge axis
    r_end = 1.25 * dp + 2.0
    for side, sx in (("L", -1), ("R", 1)):
        xa, xb_ = sorted((sx * (b / 2 + lt + 0.75), sx * (b / 2 + 2 * lt + 0.25)))
        tongue = _box(xa, xb_, yh, y0 + 3 * r_end, zd, zh) + bd.Pos((xa + xb_) / 2, yh, zh) * bd.Rot(0, 90, 0) * bd.Cylinder(r_end, xb_ - xa)
        tongue = tongue - _xcyl(dp / 2 + 0.05, xb_ - xa + 2, (xa + xb_) / 2, yh, zh)
        parts.append((f"Hinge_Tongue_{side}", tongue, "titanium_frame"))
        xs = sorted((sx * (b / 2 - 1.0), sx * (b / 2 + 3 * lt + 2.0)))
        pin = _xcyl(dp / 2, xs[1] - xs[0], (xs[0] + xs[1]) / 2, yh, zh) + _xcyl(dp / 2 + 1.5, 1.5, xs[1] if sx > 0 else xs[0], yh, zh)
        parts.append((f"Hinge_Pin_{side}", pin, "steel_shaft"))
    so, rd = g["actuator_stator_od"], g["actuator_rotor_d"]
    xa0 = W / 2 + 5.0                                        # outboard of the deck edge: the housing is larger than the hinge height
    l_act = rd + so                                          # rotor stack (L/D = 1) + strain-wave gear (~ stator OD)
    x_s0 = b / 2 + 3 * lt + 1.0
    parts.append(("Deploy_Torque_Shaft", _xcyl(8.0, xa0 - x_s0, (x_s0 + xa0) / 2, yh, zh), "steel_shaft"))
    act = _xcyl(so / 2 + 3.0, l_act, xa0 + l_act / 2, yh, zh)
    act = bd.fillet(act.edges(), radius=2.0)
    parts.append(("Deploy_Actuator_BLDC_StrainWave", act, "bldc_emerald"))
    parts.append(("Deploy_Actuator_Band", _xcyl(so / 2 + 3.6, 8.0, xa0 + 0.35 * l_act, yh, zh) - _xcyl(so / 2 + 2.0, 10.0, xa0 + 0.35 * l_act, yh, zh),
                  "copper_winding"))
    parts.append(("Deploy_Actuator_Mount", _box(xa0 - 6.0, xa0, yh - so / 2, y0 + so / 2 + 20.0, zd - td, zh + so / 2), "dark_anodized"))

    # Latch posts on the deck horns beside the spine (braced aft and outboard), spring latch pins
    for side, sx in (("L", -1), ("R", 1)):
        x = sx * xp
        post = _box(x - 20, x + 20, ysc - 20, ysc + 20, zd, zl + 30, r=4.0)
        post = post - _xcyl(g["latch_pin_d"] / 2 + 0.1, 42, x, ysc, zl)
        parts.append((f"Latch_Post_{side}", post, "titanium_frame"))
        hl = zl - zd
        parts.append((f"Latch_Post_Brace_Aft_{side}", _rod((x, ysc + 15, zl), (x, ysc + hl, zd + 4), 7.0), "dark_anodized"))
        xo = sx * min(xp + hl, W / 2 - 15)
        parts.append((f"Latch_Post_Brace_Out_{side}", _rod((x + sx * 15, ysc, zl), (xo, ysc, zd + 4), 7.0), "dark_anodized"))
        xs = sorted((sx * (xp - 21.0 - 10.0), sx * (xp + 20.0)))
        parts.append((f"Latch_Pin_{side}", _xcyl(g["latch_pin_d"] / 2, xs[1] - xs[0], (xs[0] + xs[1]) / 2, ysc, zl), "steel_shaft"))
        parts.append((f"Latch_Spring_Housing_{side}", _xcyl(9.0, 18.0, x + sx * 29.0, ysc, zl), "cam_crimson"))

    # Hold-down pedestals under the stowed spine, SMA frangible-bolt actuators under the deck
    for k, ys in enumerate(g["y_stow"]):
        ped = _cyl(16.0, zh - zd - 2.0, zd, 0, ys) - (bd.Pos(0, ys, zh - 2.0 - (g["stow_gap"] - 2.0)) * bd.Cone(7.0, 14.0, g["stow_gap"] - 1.9, align=ALIGN_BASE))
        parts.append((f"Hold_Down_Pedestal_{k + 1}", ped, "dark_anodized"))
        parts.append((f"SMA_Frangibolt_{k + 1}", _cyl(15.0, 32.0, zd - td - 32.0, 0, ys), "cam_crimson"))

    # Legs: main strut + brace from the deck edge to each footpad
    xl = g["x_leg"]
    for idx, (sx, yf, yc) in enumerate(((-1, g["y_leg_near"], y0 + 40), (1, g["y_leg_near"], y0 + 40),
                                         (-1, g["y_leg_far"], yh + L - 40), (1, g["y_leg_far"], yh + L - 40))):
        foot = (sx * xl, yf, 12.0)
        top = (sx * (W / 2 - 40), yc, zd - td)
        brace_top = (sx * (W / 2 - 40), yc + (250.0 if idx < 2 else -250.0), zd - td)
        parts.append((f"Leg_{idx + 1}_Main", _rod(top, foot, 14.0), "dark_anodized"))
        parts.append((f"Leg_{idx + 1}_Brace", _rod(brace_top, foot, 8.0), "dark_anodized"))
        pad = _cyl(g["footpad_r"], 12.0, 0.0, foot[0], foot[1])
        pad = bd.fillet(pad.edges().group_by(bd.Axis.Z)[-1], radius=4.0)
        parts.append((f"Leg_{idx + 1}_Footpad", pad, "titanium_frame"))
    return parts


def stow_location(d: Dict[str, Any]) -> bd.Location:
    """Rigid motion of the mast group from deployed to stowed: -90 deg about the hinge axis (top swings aft)."""
    g = geometry(d)
    return (bd.Pos(0, g["y_hinge"], g["z_hinge"]) * bd.Rot(-90, 0, 0) * bd.Pos(0, -g["y_hinge"], -g["z_hinge"]))


def brush_stow_location(p: Dict[str, Any], d: Dict[str, Any], by: float, z_pivot: float) -> bd.Location:
    """Brush arm swung to its launch-lock angle about the pivot pins (axle moves away from the auger)."""
    ang = d["mast"]["geometry_m"]["brush_stow_deg"]
    return (bd.Pos(0, by, z_pivot) * bd.Rot(ang, 0, 0) * bd.Pos(0, -by, -z_pivot))
