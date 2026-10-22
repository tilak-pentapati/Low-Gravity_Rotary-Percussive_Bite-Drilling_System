"""
Auger string above the bit joint - detailed build123d geometry (mm, Z-up, auger on the Z axis).

Every size comes from config/system_params.yaml (auger_geometry, auger_string, drill_head) or from the
derivations in src/design.py / src/design_string.py (transport pitch, flight attachment, braze lap,
hex shank, retention pin, guide bushing).

Parts (bit tip at z = tip, carriage underside at z = tip + auger_length):
  Transport_Core_Tube      14/12 mm tube, ends chamfered inside so the thermocouple lead is not cut
  Transport_Helix_Flight   optimum-angle flight, machined integral with the tube, root fillet r = t/2
                           (the derivation rejects fillet-welded flights on weld-toe fatigue)
  Hex_Drive_Shank          16 mm A/C hex + round collar; tube brazed into the collar socket, the tube end
                           bears on the socket floor so the blow never loads the braze; Ø6 lead bore
  Retention_Spring_Pin     cross pin through the anvil nose and the hex (carries the hanging string)
  Thermocouple_Lead        bit probe -> coaxial contact at the bit joint -> tube -> hex -> spindle -> slip ring
  TC_Coax_Contact          spring-loaded coaxial contact pair: the bit screws on without twisting the lead
  Lead_Exit_Grommet        PTFE grommet in the chamfered hex bore exit
Guide bushing (frame): Ti boss below the mast foot-plate top with a Vespel SP-3 liner and a 45 deg lead-in.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import build123d as bd

from src.cad_bit import _cyl, _helix_sweep, bit_top_z

PartT = tuple


def _tube(ro: float, ri: float, h: float, z0: float, x: float = 0.0, y: float = 0.0) -> bd.Part:
    return _cyl(ro, h, z0, x, y) - _cyl(ri, h + 0.2, z0 - 0.1, x, y)


def flight_profile(u0: float, u_tip: float, t: float, r_f: float, n_arc: int = 3) -> List[tuple]:
    """Closed (radial, axial) flight section: root fillets r_f blend into the tube surface at u0, broken tip."""
    lower = [(u0 + r_f + r_f * math.cos(math.radians(a)), -t / 2 - r_f + r_f * math.sin(math.radians(a)))
             for a in [180 - 90 * k / n_arc for k in range(n_arc + 1)]]
    upper = [(x, -y) for x, y in reversed(lower)]
    tip = [(u_tip - 0.2, -t / 2), (u_tip, -t / 2 + 0.2), (u_tip, t / 2 - 0.2), (u_tip - 0.2, t / 2)]
    return [(0.0, -t / 2 - r_f)] + lower + tip + upper + [(0.0, t / 2 + r_f)]


def build_string(p: Dict[str, Any], d: Dict[str, Any], tip: float) -> List[PartT]:
    """[(name, shape, material)] for the string between the bit joint coupling and the anvil."""
    a = p["auger_geometry"]
    s_cfg = p["auger_string"]
    ds = d["string"]
    sh = ds["shank"]
    r_od = a["outer_diameter_m"] * 1e3 / 2
    r_to, r_ti = a["tube_outer_diameter_m"] * 1e3 / 2, a["tube_inner_diameter_m"] * 1e3 / 2
    r_sb = p["drill_head"]["spindle_bore_m"] * 1e3 / 2
    zc = tip + p["cad_layout"]["auger_length_m"] * 1e3
    lap = sh["lap_m"] * 1e3
    hex_r = sh["hex_r_m"] * 1e3
    hex0 = zc - 40.0
    col0 = hex0 - 2 * lap
    z_t0 = bit_top_z(p, d, tip)
    z_t1 = hex0 - lap
    parts: List[PartT] = []

    # Tube with internal entry chamfers (lead protection)
    tube = _tube(r_to, r_ti, z_t1 - z_t0, z_t0)
    inner = tube.edges().filter_by(bd.GeomType.CIRCLE).filter_by(lambda e: abs(e.radius - r_ti) < 1e-3)
    tube = bd.chamfer(inner, length=0.3)
    parts.append(("Transport_Core_Tube", tube, "steel_shaft"))

    # Integral flight with root fillets
    t_tr = a["transport_flight_thickness_m"] * 1e3
    pitch = d["auger"]["transport_pitch_mm"]
    r_h = r_to - 0.3
    prof = flight_profile(0.3, r_od - r_h, t_tr, ds["flight"]["root_fillet_m"] * 1e3)
    z_f0 = z_t0 + 2.0 + ds["flight"]["root_fillet_m"] * 1e3
    parts.append(("Transport_Helix_Flight", _helix_sweep(prof, r_h, pitch, (col0 - 3.0) - z_f0, z_f0), "flute_hardened"))

    # Hex shank + collar (socket for the tube, lead bore, cross-pin hole)
    hx = bd.Pos(0, 0, hex0) * bd.extrude(bd.RegularPolygon(hex_r, 6), 80.0)
    hx = bd.chamfer(hx.edges().filter_by(bd.Axis.Z, reverse=True).group_by(bd.Axis.Z)[-1], length=0.5)
    col = _cyl(s_cfg["shank_collar_od_m"] * 1e3 / 2, 2 * lap, col0)
    col = bd.chamfer(col.edges().filter_by(bd.GeomType.CIRCLE).sort_by(bd.Axis.Z)[0], length=0.8)
    shank = hx + col
    shank = shank - _cyl(r_to + 0.05, lap, col0 - 0.01) - _cyl(r_sb, 2 * lap + 82.0, col0 - 1.0)
    shank = shank - bd.Pos(0, 0, hex0 + 80.0 - 0.8) * bd.Cone(r_sb, r_sb + 0.8, 0.81, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    z_pin = zc + sh["pin_z_mm"]
    d_pin = sh["pin_d_m"] * 1e3
    shank = shank - bd.Pos(0, 0, z_pin) * bd.Rot(0, 90, 0) * bd.Cylinder(d_pin / 2 + 0.02, 40.0)
    parts.append(("Hex_Drive_Shank", shank, "steel_shaft"))
    parts.append(("Retention_Spring_Pin", bd.Pos(0, 0, z_pin) * bd.Rot(0, 90, 0) * bd.Cylinder(d_pin / 2, 29.6), "gearbox_gunmetal"))
    parts.append(("Shank_Braze_Fillet", _tube(r_to + 0.6, r_to, 0.6, col0 - 0.6), "brass_gear"))

    # Thermocouple lead: bit probe top -> coax contact at the joint -> up the string -> spindle bore
    b = p["drill_bit"]
    pr = a["cutter_protrusion_m"] * 1e3
    z_probe = tip + pr + b["crown_height_m"] * 1e3 + 40.0
    z_sh = tip + pr + b["crown_height_m"] * 1e3 + d["auger"]["sampling_length_mm"]
    z_c0, z_c1 = z_sh + 2.0, z_sh + 10.0
    r_l = ds["lead_od_m"] * 1e3 / 2
    parts.append(("TC_Coax_Contact", _cyl(a["wiring_bore_m"] * 1e3 / 2 - 0.2, z_c1 - z_c0, z_c0), "gold_kapton"))
    lead = _cyl(r_l, z_c0 - z_probe, z_probe) + _cyl(r_l, zc + 40.0 - z_c1, z_c1)
    parts.append(("Thermocouple_Lead", lead, "copper_winding"))
    parts.append(("Lead_Exit_Grommet", _tube(r_sb - 0.05, r_l + 0.1, 4.0, zc + 40.0 - 4.5), "radiator_white"))
    return parts


def build_guide_bushing(p: Dict[str, Any], d: Dict[str, Any]) -> List[PartT]:
    """Mast-foot guide bushing: Ti boss (flush with the foot-plate top) + Vespel SP-3 liner with a 45 deg lead-in."""
    bu = d["string"]["bushing"]
    r_b = bu["bore_m"] * 1e3 / 2
    lead = bu["leadin_m"] * 1e3
    z0 = bu["z0_m"] * 1e3
    h = bu["length_m"] * 1e3
    dt = p["cad_layout"]["foot_top_m"] * 1e3
    liner = _tube(r_b + 1.5, r_b, h, z0)
    liner = liner - bd.Pos(0, 0, z0 + h - lead) * bd.Cone(r_b, r_b + lead, lead + 0.01, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    boss = _tube(r_b + 4.5, r_b + 1.5, dt - z0, z0) + _tube(r_b + 9.0, r_b + 1.5, 3.0, dt - p["drill_mast"]["foot_plate_thickness_m"] * 1e3 - 3.0)  # flange under the foot plate
    return [("Guide_Bushing_Liner_SP3", liner, "vespel_brown"), ("Guide_Bushing_Boss", boss, "dark_anodized")]


def boss_radius(d: Dict[str, Any]) -> float:
    """Outer radius (mm) of the guide-bushing boss = foot-plate hole radius."""
    return d["string"]["bushing"]["bore_m"] * 1e3 / 2 + 4.5
