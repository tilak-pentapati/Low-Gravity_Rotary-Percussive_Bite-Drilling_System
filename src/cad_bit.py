"""
Drill-bit crown, sampling sub and joint - detailed build123d geometry (mm, Z-up).

Every dimension comes from config/system_params.yaml (drill_bit, auger_geometry) or from the
derivations in src/design.py (edge radius, junk-slot width, flute depth, pitches, lengths).

Parts (bit tip at z = tip):
  WC_Insert_1/2          two chisel inserts, -10 deg rake / 15 deg clearance, Hertz-sized edge radius,
                         cutting edges on the line through the axis (centre is cut), 2 mm proud of the face
  Bit_Crown_Body         17-4PH body with brazed insert pockets, two junk slots ahead of each rake face,
                         face chamfer and a thermocouple pocket ending just above the inserts
  Gauge_Reamer_Pad_1/2   carbide pads on the trailing quadrants with back-reaming chamfers
  Thermocouple_Probe     type-T sheath in the pocket, leads up the hollow string to the slip ring
  Sampling_Root_Shaft    fatigue-sized hollow root + Sampling_Flight
  Bit_Sub_Shoulder       impact-bearing shoulder (impact passes through it, not the thread)
  Bit_Sub_Pin_Thread     M12x1 pin, right-hand (drilling torque tightens it)
  Joint_Coupling         female coupling brazed onto the transport tube
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import build123d as bd

ALIGN_BASE = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)


def _cyl(r: float, h: float, z0: float, x: float = 0.0, y: float = 0.0) -> bd.Part:
    return bd.Pos(x, y, z0) * bd.Cylinder(r, h, align=ALIGN_BASE)


def _helix_sweep(profile_pts, radius: float, pitch: float, height: float, z0: float) -> bd.Part:
    """Sweep a closed (radial, axial) profile along a right-hand helix (threads, flights)."""
    helix = bd.Helix(pitch=pitch, height=height, radius=radius)
    plane = bd.Plane(origin=helix @ 0, x_dir=(1, 0, 0), z_dir=helix % 0)
    with bd.BuildPart() as bp:
        with bd.BuildLine():
            bd.add(helix)
        with bd.BuildSketch(plane):
            bd.Polygon(*profile_pts, align=None)
        bd.sweep(is_frenet=True)
    return bd.Pos(0, 0, z0) * bp.part


def insert_solid(p: Dict[str, Any], d: Dict[str, Any]) -> bd.Part:
    """One chisel insert for the +X wing (cuts while moving -Y), tip edge on the X axis at z = 0."""
    b = p["drill_bit"]
    t = b["insert_thickness_m"] * 1e3
    h = b["insert_height_m"] * 1e3
    rake = math.radians(b["rake_angle_deg"])
    clr = math.radians(b["clearance_angle_deg"])
    r_e = d["bit"]["edge_radius_m"] * 1e3
    gauge_r = p["auger_geometry"]["gauge_diameter_m"] * 1e3 / 2
    length = gauge_r - 0.5
    # Section in the (y, z) plane: tip, flank (clearance) to the back, back face, top, rake face to the tip.
    # Negative rake: the rake face leans forward (toward -y) as it rises.
    pts = [(0.0, 0.0), (t, t * math.tan(clr)), (t, h), (h * math.tan(rake), h)]
    with bd.BuildPart() as bp:
        with bd.BuildSketch(bd.Plane.YZ):
            bd.Polygon(*pts, align=None)
        bd.extrude(amount=length)
        tip_edge = bp.edges().filter_by(bd.Axis.X).sort_by(bd.Axis.Z)[0]
        bd.fillet(tip_edge, radius=r_e)
        gauge_edges = bp.faces().sort_by(bd.Axis.X)[-1].edges()
        bd.chamfer(gauge_edges, length=0.3)
    return bd.Pos(0.5, 0, 0) * bp.part


def build_bit(p: Dict[str, Any], d: Dict[str, Any], tip: float) -> List[tuple]:
    """Returns [(name, shape, material)] for the bit crown, sampling sub and joint."""
    b = p["drill_bit"]
    a = p["auger_geometry"]
    db = d["bit"]
    da = d["auger"]
    pr = a["cutter_protrusion_m"] * 1e3
    crown_h = b["crown_height_m"] * 1e3
    h_ins = b["insert_height_m"] * 1e3
    t_ins = b["insert_thickness_m"] * 1e3
    body_r = db["body_r_m"] * 1e3
    r_od = a["outer_diameter_m"] * 1e3 / 2
    gauge_r = a["gauge_diameter_m"] * 1e3 / 2
    bore_r = a["wiring_bore_m"] * 1e3 / 2
    slot_w = db["slot_width_m"] * 1e3
    slot_in = db["slot_inner_m"] * 1e3
    parts: List[tuple] = []

    ins = insert_solid(p, d)
    inserts = [bd.Pos(0, 0, tip) * ins, bd.Pos(0, 0, tip) * bd.Rot(0, 0, 180) * ins]
    for i, s in enumerate(inserts):
        parts.append((f"WC_Insert_{i + 1}", s, "tungsten_carbide"))

    # Crown body
    z_face = tip + pr
    body = _cyl(body_r, crown_h, z_face)
    body = bd.chamfer(body.edges().sort_by(bd.Axis.Z)[0], length=0.8)
    for s in inserts:  # brazed pockets (braze gap is below CAD resolution)
        body = body - s
    for sy in (-1, 1):  # junk slots ahead of each rake face
        body = body - bd.Pos(0, sy * (slot_in + 0.5 * (body_r + 1 - slot_in)), z_face + crown_h / 2) * bd.Box(
            slot_w, body_r + 1 - slot_in, crown_h + 2)
    # Thermocouple pocket: wiring bore down to just above the inserts
    pocket_bottom = tip + h_ins + b["thermocouple_standoff_m"] * 1e3
    body = body - _cyl(bore_r, z_face + crown_h - pocket_bottom + 1, pocket_bottom)
    parts.append(("Bit_Crown_Body", body, "gearbox_gunmetal"))

    # Gauge reamer pads on the trailing quadrants, with back-reaming chamfers on top
    for i, ang in enumerate((45.0, 225.0)):
        pad = bd.Pos(0.5 * (body_r - 0.6 + gauge_r - 0.1), 0, z_face + crown_h - 7.0) * bd.Box(gauge_r - 0.1 - (body_r - 0.6), 5.0, 8.0)
        pad = bd.chamfer(pad.edges().filter_by(bd.Axis.Y).sort_by(bd.Axis.Z)[-1:], length=1.0)
        parts.append((f"Gauge_Reamer_Pad_{i + 1}", bd.Rot(0, 0, ang) * pad, "tungsten_carbide"))

    # Thermocouple probe (sheath) sitting on top of the inserts
    tc_r = b["thermocouple_bore_m"] * 1e3 / 2 * 0.8
    parts.append(("Thermocouple_Probe", _cyl(tc_r, z_face + crown_h + 40 - pocket_bottom, pocket_bottom + 0.1), "copper_winding"))

    # Sampling section: hollow root + flight
    root_r = da["root_od_mm"] / 2
    z_s0 = z_face + crown_h
    l_s = da["sampling_length_mm"]
    root = _cyl(root_r, l_s, z_s0) - _cyl(bore_r, l_s + 2, z_s0 - 1)
    parts.append(("Sampling_Root_Shaft", root, "steel_shaft"))
    pitch = da["sampling_pitch_mm"]
    t_f = a["sampling_flight_thickness_m"] * 1e3
    flight = _helix_sweep([(0, -t_f / 2), (r_od - root_r + 0.3, -t_f / 2), (r_od - root_r + 0.3, t_f / 2), (0, t_f / 2)],
                          root_r - 0.3, pitch, l_s - 2.0, z_s0 + 1.0)
    parts.append(("Sampling_Flight", flight, "flute_hardened"))

    # Joint: impact shoulder + M12x1 right-hand pin + female coupling on the transport tube
    sh_r = b["joint_shoulder_od_m"] * 1e3 / 2
    d_maj = b["joint_thread_major_m"] * 1e3
    p_th = b["joint_thread_pitch_m"] * 1e3
    l_e = b["joint_engagement_m"] * 1e3
    z_sh = z_s0 + l_s
    shoulder = _cyl(sh_r, 6.0, z_sh) - _cyl(bore_r, 8.0, z_sh - 1)
    shoulder = bd.chamfer(shoulder.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.Z)[0], length=1.5)
    parts.append(("Bit_Sub_Shoulder", shoulder, "steel_shaft"))
    r_minor = d_maj / 2 - 0.6134 * p_th
    pin = _cyl(r_minor, l_e, z_sh + 6.0) - _cyl(bore_r, l_e + 2, z_sh + 5.0)
    h_t = 0.6134 * p_th
    thread = _helix_sweep([(0, -0.45 * p_th), (h_t, -0.06 * p_th), (h_t, 0.06 * p_th), (0, 0.45 * p_th)],
                          r_minor - 0.05, p_th, l_e - 1.5, z_sh + 6.5)
    parts.append(("Bit_Sub_Pin_Thread", pin + thread, "steel_shaft"))
    coupling = _cyl(sh_r, l_e + 4.0, z_sh + 6.0) - _cyl(d_maj / 2 + 0.05, l_e + 0.5, z_sh + 5.9)
    coupling = coupling - _cyl(a["tube_outer_diameter_m"] * 1e3 / 2, 4.5, z_sh + 6.0 + l_e)
    parts.append(("Joint_Coupling", coupling, "brass_gear"))
    return parts


def bit_top_z(p: Dict[str, Any], d: Dict[str, Any], tip: float) -> float:
    """z where the transport tube starts (top of the joint coupling)."""
    a, b = p["auger_geometry"], p["drill_bit"]
    return (tip + a["cutter_protrusion_m"] * 1e3 + b["crown_height_m"] * 1e3 + d["auger"]["sampling_length_mm"]
            + 6.0 + b["joint_engagement_m"] * 1e3 + 4.0 - 4.5)


EXPLODE = {  # (dx, dy, dz) offsets in mm for the exploded view
    "WC_Insert_1": (14, -10, -28), "WC_Insert_2": (-14, 10, -28),
    "Gauge_Reamer_Pad_1": (12, 12, 0), "Gauge_Reamer_Pad_2": (-12, -12, 0),
    "Thermocouple_Probe": (30, 0, 25), "Bit_Crown_Body": (0, 0, 0),
    "Sampling_Root_Shaft": (0, 0, 20), "Sampling_Flight": (0, 0, 20),
    "Bit_Sub_Shoulder": (0, 0, 35), "Bit_Sub_Pin_Thread": (0, 0, 35), "Joint_Coupling": (0, 0, 60),
}
