"""
Module 3: Programmatic 3D CAD Generation (build123d B-rep -> STEP / STL / GLB).

Every part is modelled ONCE as a true build123d B-rep solid (millimetres, Z-up).
The same solids are then:
  * written to colour-labelled, assembly-structured ISO-10303-21 .step files (mm),
  * tessellated into .stl meshes (mm), and
  * tessellated into glTF 2.0 .glb scenes (metres, Y-up) for <model-viewer>/Three.js.
so the engineering STEP and the web models can never drift apart.

Assemblies exported to exports/cad/ (GLBs are also synced to docs_src/assets/cad/):
  1. auger_bit        - 20 mm OD dual-zone auger with WC bit crown
  2. drill_bit_crown  - close-up of the bit crown + 10 cm sampling-flute section
  3. drill_head       - enclosed rotary-percussive drill head (src/cad_head.py) + quarter-section view
  4. mast_and_brush   - deployed mast (src/cad_mast.py) on the lander deck interface, Z-feed and brushing station
     mast_stowed      - launch configuration: mast rotated about the hinge onto the hold-downs, carriage at the stow pose
  5. power_system     - 4 x 0.68 m^2 Ge solar wings, avionics warm box, H2/O2 tanks
  6. full_drill_rig   - everything, with named kinematic parent nodes:
       Frame_Node, Solar_Node, Carriage_Node, Auger_Node, Hammer_Cam_Node, Cam_Ring_Node, Brush_Gear_Node

The full rig is modelled in its HOME pose (bit tip at cad_layout.home_tip_height_m).
The digital twin moves Carriage_Node / Auger_Node / Hammer_Cam_Node vertically by
(tip_target - home_tip) and spins Auger_Node / Brush_Gear_Node about their own axes.

Coordinates:
  build123d / STEP : X right, Y aft (rear), Z up, millimetres, z = 0 at the regolith surface
  glTF / Three.js  : X right, Y up, Z fore (toward the brushing station), metres
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

import build123d as bd

from src.design import derive_all, load_system_params
from src.cad_bit import EXPLODE as BIT_EXPLODE, bit_top_z, build_bit
from src.cad_head import COARSE_MESH, build_head, section as head_section
from src.cad_auger import boss_radius, build_guide_bushing, build_string
from src.cad_brush import (build_brush_frame, build_brush_local as brush_local_parts, build_carousel_local as carousel_local_parts,
                           explode_offset, geometry as brush_geometry)
from src.cad_mast import brush_stow_location, build_lander, build_mast, geometry as mast_geometry, stow_location


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_CAD_DIR = PROJECT_ROOT / "exports" / "cad"
EXPORTS_DATA_DIR = PROJECT_ROOT / "exports" / "data"
DOCS_CAD_DIR = PROJECT_ROOT / "docs_src" / "assets" / "cad"

ALIGN_BASE = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)

# Z-up millimetres -> Y-up metres:  (x, y, z)_bd  ->  (x, z, -y)_gltf * 1e-3
BD_TO_GLTF = np.array(
    [
        [1e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-3, 0.0],
        [0.0, -1e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ==============================================================================
# FUNCTIONAL COLOUR CODE
# Parts are coloured by the job they do in the machine (load path / energy path), not by a
# guessed surface finish. Hues are the Okabe-Ito colour-universal-design set, so the eight
# classes stay distinct for colour-vision-deficient readers; finishes are matte so the hue,
# not specular highlights, carries the information. First matching pattern wins.
# (key, label, function, RGBA 0..255, metallicFactor, roughnessFactor, name pattern)
# ==============================================================================
COLOUR_CODE: List[Tuple[str, str, str, Tuple[int, int, int, int], float, float, str]] = [
    ("cutting", "Rock-cutting interface", "Tungsten-carbide inserts and gauge pads: the only parts that touch intact rock",
     (38, 38, 38, 255), 0.30, 0.40, r"WC_Insert|Gauge_Reamer"),
    ("tribology", "Seals, bushings &amp; dust control", "PTFE / Vespel sliding contacts, lip seals, labyrinths, bellows and wipers that keep regolith out",
     (204, 121, 167, 255), 0.05, 0.65, r"PTFE|Lip_Seal|Labyrinth|Bellows|Wiper|Liner_SP|Axle_Bushing|Grommet|Cup_Seal|Diaphragm"),
    ("electrical", "Electromechanical conversion &amp; sensing", "BLDC motors, encoders, slip ring, load cell, thermocouple path and solar cells",
     (0, 114, 178, 255), 0.15, 0.55, r"BLDC|Motor|Encoder|Slip_Ring|Thermocouple|TC_Coax|GaInP|Busbar|Load_Cell|Actuator_Band"),
    ("percussion", "Percussive (impact) path", "Barrel cam, followers, hammer, spring and anvil: stores and releases the blow energy",
     (213, 94, 0, 255), 0.10, 0.55, r"Hammer|Anvil|Barrel_Cam|Cam_Follower|Cam_Bearing"),
    ("deployment", "Deployment &amp; launch restraint", "Hinge, latch, hold-down and release hardware that only works before drilling",
     (86, 180, 233, 255), 0.10, 0.55, r"Hinge|Latch|Hold_Down|Frangibolt|Launch_Lock|Deploy_Torque"),
    ("sample", "Drill string &amp; sample path", "Every part that carries or touches the regolith sample: bit body, auger, brush, chute, cups",
     (0, 158, 115, 255), 0.10, 0.55, r"Flight|Crown_Body|Sampling_Root|Transport_Core|Bit_Sub|Joint_Coupling|Hex_Drive|Shank_Braze|"
                                     r"Retention_Spring|Brush_|Bristle|Chute|Funnel|Sample_Cup|Carousel"),
    ("drive", "Rotary &amp; linear drive train", "Gears, spindle, shafts, bearings, leadscrew, nut and rails that transmit torque and feed force",
     (230, 159, 0, 255), 0.15, 0.50, r"Gear|Pinion|Planet|Spindle_Bearing_7|Splined|Leadscrew|Nut|Bearing_6|"
                                     r"_Balls|_Rings|Shaft|Guide_Rail|LM16UU|Elevation_Pin|Platter|Elevation_Bearing"),
    ("thermal", "Thermal control &amp; energy storage", "Avionics warm box, radiator and H<sub>2</sub>/O<sub>2</sub> reactant tanks",
     (240, 228, 66, 255), 0.05, 0.60, r"WarmBox|Radiator|Reactant_Tank"),
    ("lander", "Lander interface (context)", "Deck section, legs and footpads that the drill reacts against",
     (122, 118, 110, 255), 0.05, 0.75, r"Lander_Deck|Deck_Face|Leg_\d|Footpad"),
    ("structure", "Primary structure", "Housings, mast spine, plates, brackets, frames and clamps that carry the loads",
     (190, 184, 170, 255), 0.10, 0.60, r".*"),
]
_COLOUR_RE = [(k, re.compile(pat)) for k, _, _, _, _, _, pat in COLOUR_CODE]
_COLOUR = {k: (rgba, met, rough) for k, _, _, rgba, met, rough, _ in COLOUR_CODE}


def _srgb_to_linear(c8: int) -> float:
    """glTF baseColorFactor is linear RGB; the code above is in display (sRGB) values (IEC 61966-2-1)."""
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def colour_class(part_name: str) -> str:
    name = part_name.removeprefix("Mesh_Context_")
    return next(k for k, rx in _COLOUR_RE if rx.search(name))


@dataclass
class Part:
    name: str
    shape: bd.Shape
    material: str


def _color(part_name: str) -> bd.Color:
    rgba = _COLOUR[colour_class(part_name)][0]
    return bd.Color(rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0)


# ==============================================================================
# PRIMITIVE HELPERS (all mm, Z-up)
# ==============================================================================
def cyl(r: float, h: float, x: float = 0.0, y: float = 0.0, z0: float = 0.0) -> bd.Part:
    """Vertical cylinder with its base at z0."""
    return bd.Pos(x, y, z0) * bd.Cylinder(r, h, align=ALIGN_BASE)


def tube(ro: float, ri: float, h: float, x: float = 0.0, y: float = 0.0, z0: float = 0.0) -> bd.Part:
    """Vertical hollow tube with its base at z0."""
    return cyl(ro, h, x, y, z0) - cyl(ri, h + 0.2, x, y, z0 - 0.1)


def box(dx: float, dy: float, dz: float, cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> bd.Part:
    """Axis-aligned box centred at (cx, cy, cz)."""
    return bd.Pos(cx, cy, cz) * bd.Box(dx, dy, dz)


def rod(p1: Sequence[float], p2: Sequence[float], r: float) -> bd.Solid:
    """Cylindrical strut between two points."""
    a, b = bd.Vector(*p1), bd.Vector(*p2)
    return bd.Solid.make_cylinder(r, (b - a).length, bd.Plane(origin=a, z_dir=b - a))


def x_cyl(r: float, length: float, cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> bd.Part:
    """Cylinder whose axis is the X axis, centred at (cx, cy, cz)."""
    return bd.Pos(cx, cy, cz) * bd.Rot(0, 90, 0) * bd.Cylinder(r, length)


def helical_flight(
    r_root: float, r_od: float, z0: float, height: float, pitch: float, thickness: float, phase_deg: float = 0.0
) -> bd.Part:
    """Swept rectangular auger flight (right-hand helix) from r_root to r_od."""
    helix = bd.Helix(pitch=pitch, height=height, radius=r_root)
    plane = bd.Plane(origin=helix @ 0, x_dir=(1, 0, 0), z_dir=helix % 0)
    with bd.BuildPart() as p:
        with bd.BuildLine():
            bd.add(helix)
        with bd.BuildSketch(plane):
            bd.Rectangle(r_od - r_root + 0.3, thickness, align=(bd.Align.MIN, bd.Align.CENTER))
        bd.sweep(is_frenet=True)
    return bd.Pos(0, 0, z0) * bd.Rot(0, 0, phase_deg) * p.part


def coil_spring(coil_r: float, wire_r: float, z0: float, height: float, turns: float) -> bd.Part:
    """Helical compression spring (circular wire swept along a helix)."""
    helix = bd.Helix(pitch=height / turns, height=height, radius=coil_r)
    plane = bd.Plane(origin=helix @ 0, x_dir=(1, 0, 0), z_dir=helix % 0)
    with bd.BuildPart() as p:
        with bd.BuildLine():
            bd.add(helix)
        with bd.BuildSketch(plane):
            bd.Circle(wire_r)
        bd.sweep(is_frenet=True)
    return bd.Pos(0, 0, z0) * p.part


def spur_gear(r_pitch: float, teeth: int, h: float, x: float = 0.0, y: float = 0.0, z0: float = 0.0,
              bore: float = 0.0) -> bd.Part:
    """Visual spur gear: root cylinder plus rectangular teeth on the pitch circle."""
    module = 2.0 * r_pitch / teeth
    root = r_pitch - 1.25 * module
    g = cyl(root, h, 0, 0, 0)
    tooth_w = 0.5 * math.pi * module
    for i in range(teeth):
        ang = 360.0 * i / teeth
        g = g + bd.Rot(0, 0, ang) * box(2.25 * module, tooth_w, h, root + 1.0 * module, 0, 0.5 * h)
    if bore > 0:
        g = g - cyl(bore, h + 1, 0, 0, -0.5)
    return bd.Pos(x, y, z0) * g


# ==============================================================================
# LAYOUT (derived from config/system_params.yaml)
# ==============================================================================
@dataclass
class Layout:
    od: float
    id_tube: float
    od_tube: float
    flute_depth: float
    sample_len: float
    sample_pitch: float
    upper_pitch: float
    auger_len: float
    home_tip: float
    foot_top: float
    brush_z: float
    brush_r: float
    root_od: float
    bore: float
    t_samp: float
    t_tr: float
    edge_r: float
    protrusion: float
    sweep_mm: float
    wing_w: float
    wing_h: float
    leadscrew_lead: float
    # Derived drill-head / mast stack (src/design_head.py, src/design_mast.py)
    carriage_h: float = 340.0
    rail_x: float = 90.0
    leadscrew_y: float = 62.0
    wing_sep: float = 0.0
    wing_y: float = 0.0
    wing_z0: float = 0.0
    deck_top: float = 0.0
    brush_y: float = 0.0
    params: Optional[dict] = None
    design: Optional[dict] = None

    @property
    def zc_home(self) -> float:
        return self.home_tip + self.auger_len

    @property
    def mast_top(self) -> float:
        return self.design["mast"]["geometry_m"]["z_top"] * 1e3


def load_layout(params: Optional[dict] = None, design: Optional[dict] = None) -> Layout:
    """Geometry from config primary inputs + src/design.py derivations (all in mm)."""
    params = params or load_system_params()
    design = design or derive_all(params)
    a = params["auger_geometry"]
    c = params["cad_layout"]
    sa = params["power_system"]["solar_array"]
    dz = design["auger"]
    od = a["outer_diameter_m"] * 1000.0
    lay = Layout(
        od=od,
        id_tube=a["tube_inner_diameter_m"] * 1000.0,
        od_tube=a["tube_outer_diameter_m"] * 1000.0,
        flute_depth=float(dz["flute_depth_mm"]),
        sample_len=float(dz["sampling_length_mm"]),
        sample_pitch=float(dz["sampling_pitch_mm"]),
        upper_pitch=float(dz["transport_pitch_mm"]),
        auger_len=c["auger_length_m"] * 1000.0,
        home_tip=c["home_tip_height_m"] * 1000.0,
        foot_top=c["foot_top_m"] * 1000.0,
        brush_z=c["brush_axle_height_m"] * 1000.0,
        brush_r=design["brush"]["mesh"]["r_p_m"] * 1000.0,
        root_od=float(dz["root_od_mm"]),
        bore=a["wiring_bore_m"] * 1000.0,
        t_samp=a["sampling_flight_thickness_m"] * 1000.0,
        t_tr=a["transport_flight_thickness_m"] * 1000.0,
        edge_r=design["bit"]["edge_radius_m"] * 1000.0,
        protrusion=a["cutter_protrusion_m"] * 1000.0,
        sweep_mm=float(dz["brush_sweep_mm_s"]),
        wing_w=sa["wing_width_m"] * 1000.0,
        wing_h=sa["wing_height_m"] * 1000.0,
        leadscrew_lead=params["actuators"]["z_axis_feed"]["lead_mm"],
    )
    gm = design["mast"]["geometry_m"]
    lay.wing_sep, lay.wing_y, lay.wing_z0 = 2e3 * gm["wing_x"], 1e3 * gm["y_cg"], 1e3 * gm["wing_z0"]
    lay.deck_top = 1e3 * gm["z_deck"]
    lay.carriage_h = params["drill_mast"]["carriage_envelope_m"] * 1e3
    lay.rail_x = params["drill_mast"]["rail_x_m"] * 1e3
    # Brush axle on the derived worm-wheel centre distance (pitch circle on the mean flute radius)
    lay.brush_y = -design["brush"]["mesh"]["axle_offset_m"] * 1000.0
    lay.leadscrew_y = float(design["head"]["plan_mm"]["leadscrew_y"])
    assert design["head"]["stack_mm"]["yoke_top"] <= lay.carriage_h, "drill head taller than the carriage envelope"
    lay.params, lay.design = params, design
    return lay


# ==============================================================================
# SUBASSEMBLY 1: AUGER + BIT CROWN
# ==============================================================================
CROWN_PART_NAMES = set(BIT_EXPLODE)


def build_auger(lay: Layout, tip: float) -> List[Part]:
    """20 mm OD dual-zone auger: WC bit crown, derived-depth sampling flutes, optimum-angle transport helix."""
    parts: List[Part] = []

    # --- Detailed bit crown, sampling sub and M12x1 joint (src/cad_bit.py)
    for name, shape, mat in build_bit(lay.params, lay.design, tip):
        parts.append(Part(name, shape, mat))

    # --- Transport tube, integral flight, hex shank, retention pin and thermocouple path (src/cad_auger.py)
    for name, shape, mat in build_string(lay.params, lay.design, tip):
        parts.append(Part(name, shape, mat))
    return parts


# ==============================================================================
# SUBASSEMBLY 2: DRILL HEAD (src/cad_head.py)
# ==============================================================================
def build_drill_head(lay: Layout, zc: float) -> Dict[str, List[Part]]:
    """Enclosed rotary-percussive head with the carriage underside at zc: carriage / spindle / hammer / cam groups."""
    groups = build_head(lay.params, lay.design, zc, lay.rail_x)
    return {k: [Part(n, sh, m) for n, sh, m in v] for k, v in groups.items()}


# ==============================================================================
# SUBASSEMBLY 3: DECK, MAST, Z-FEED, BRUSHING STATION
# ==============================================================================
def build_brush_local(lay: Layout) -> List[Part]:
    """Passive steel-bristle worm wheel (src/cad_brush.py), axle along X, centred at the LOCAL origin."""
    return [Part(n, sh, m) for n, sh, m in brush_local_parts(lay.params, lay.design)]


def build_carousel_local(lay: Layout) -> List[Part]:
    """Rotating sample carousel (src/cad_brush.py) about its own axis at the LOCAL origin."""
    return [Part(n, sh, m) for n, sh, m in carousel_local_parts(lay.params, lay.design)]


def carousel_offset(lay: Layout) -> Tuple[float, float, float]:
    return (0.0, -lay.design["brush"]["carousel"]["axis_offset_m"] * 1000.0, 0.0)


def build_frame(lay: Layout) -> List[Part]:
    """Deployed mast (src/cad_mast.py) + guide bushing + brushing-station frame parts (all rotate with the mast)."""
    parts = [Part(n, sh, m) for n, sh, m in build_mast(lay.params, lay.design)]
    parts += [Part(n, sh, m) for n, sh, m in build_guide_bushing(lay.params, lay.design)]
    # Brushing station: pivot arms, spring, bushings, chute and sample cup (src/cad_brush.py)
    parts += [Part(n, sh, m) for n, sh, m in build_brush_frame(lay.params, lay.design)]
    return parts


def build_lander_parts(lay: Layout) -> List[Part]:
    """Static lander-deck interface: deck, hinge, actuator, latch posts, hold-down pedestals, legs."""
    return [Part(n, sh, m) for n, sh, m in build_lander(lay.params, lay.design)]


def stowed_parts(lay: Layout, frame: List[Part], moving: List[Part], brush_local: List[Part],
                 carousel: Optional[List[Part]] = None) -> List[Part]:
    """Launch configuration: carriage group lowered to the stow pose, brush arm on its launch lock, mast rotated aft."""
    loc = stow_location(lay.design)
    drop = lay.design["mast"]["geometry_m"]["carriage_drop"] * 1e3
    g = brush_geometry(lay.params, lay.design)
    b_loc = brush_stow_location(lay.params, lay.design, lay.brush_y, lay.brush_z - g["l_arm"])
    swung = ("Brush_Arm", "Brush_Axle_Bushing")
    out = []
    for q in frame:
        sh = q.shape.moved(b_loc) if q.name.startswith(swung) else q.shape
        out.append(Part(q.name, sh.moved(loc), q.material))
    for q in brush_local:
        sh = bd.Pos(0, lay.brush_y, lay.brush_z) * q.shape
        out.append(Part(q.name, sh.moved(b_loc).moved(loc), q.material))
    for q in moving:
        out.append(Part(q.name, (bd.Pos(0, 0, -drop) * q.shape).moved(loc), q.material))
    for q in carousel or []:
        out.append(Part(q.name, (bd.Pos(*carousel_offset(lay)) * q.shape).moved(loc), q.material))
    return out


# ==============================================================================
# SUBASSEMBLY 4: POWER SYSTEM
# ==============================================================================
WING_SIDES = (("Port", -1.0), ("Starboard", 1.0))


def build_power(lay: Layout) -> Dict[str, Any]:
    """
    Power system: avionics warm box, 4 H2/O2 reactant tanks, and two 2-axis sun-tracking
    wings (selected by src/solar.py). Each wing = 2 panels (0.80 x 0.85 m) stacked, on an
    azimuth turntable at the end of an outrigger and an elevation yoke through the wing centre.

    Returns {"static": [...], "<side>": {"az": [...], "elev": [...], "az_origin": (x,y,z),
    "elev_origin": (0,0,h)}} with az/elev parts in their node-local frames (default pose:
    cells facing -Y / fore, vertical).
    """
    gm = mast_geometry(lay.design)
    half, dt = 0.5 * gm["deck_width"], lay.deck_top
    yw = lay.wing_y
    out: Dict[str, Any] = {"static": []}
    st = out["static"]

    # Reactant tanks (port side) and avionics warm box (starboard side) beside the stowed-mast strip, aft of the latch posts
    y_t0 = gm["y_hinge"] + (gm["z_latch"] - gm["z_deck"]) + 60.0
    x_t = half - 55.0
    for idx in range(4):
        y = y_t0 + 50.0 + 110.0 * idx
        body = cyl(45.0, 190.0, -x_t, y, dt + 45.0)
        body = body + bd.Pos(-x_t, y, dt + 45.0) * bd.Sphere(45.0) + bd.Pos(-x_t, y, dt + 235.0) * bd.Sphere(45.0)
        st.append(Part(f"H2O2_Reactant_Tank_{idx + 1}", body - box(200, 200, 100, -x_t, y, dt - 50.0), "tank_white"))
        for k, zr in enumerate((dt + 90.0, dt + 190.0)):
            st.append(Part(f"Tank_{idx + 1}_Clamp_{k + 1}", tube(47.5, 45.0, 10.0, -x_t, y, zr), "cam_crimson"))
    st.append(Part("Tank_Saddle_Rail", box(20.0, 440.0, 12.0, -x_t, y_t0 + 215.0, dt + 6.0), "dark_anodized"))
    wb = box(130.0, 420.0, 160.0, x_t - 10.0, y_t0 + 210.0, dt + 80.0)
    st.append(Part("Avionics_WarmBox", bd.fillet(wb.edges().filter_by(bd.Axis.Z), radius=6.0), "gold_kapton"))
    st.append(Part("Avionics_Radiator", box(136.0, 426.0, 6.0, x_t - 10.0, y_t0 + 210.0, dt + 163.0), "radiator_white"))

    ax = 0.5 * lay.wing_sep
    w, h = lay.wing_w, lay.wing_h
    z_axis = lay.wing_z0 + 0.5 * h
    for side, sx in WING_SIDES:
        # Outrigger: two beams from the deck edge to the turntable housing (cantilevered from the deck)
        for j, yy in enumerate((yw - 220.0, yw + 220.0)):
            st.append(Part(f"{side}_Outrigger_Beam_{j + 1}", rod((sx * (half - 20), yy, dt - 12.0), (sx * ax, yw, dt - 12.0), 9.0), "titanium_frame"))
        st.append(Part(f"{side}_Outrigger_Diagonal", rod((sx * (half - 20), yw, dt - 24.0), (sx * (ax - 60.0), yw, dt - 12.0), 7.0), "dark_anodized"))
        st.append(Part(f"{side}_Turntable_Housing", cyl(55.0, 40.0, sx * ax, yw, dt - 30.0), "gearbox_gunmetal"))
        st.append(Part(f"{side}_Azimuth_Motor", cyl(20.0, 45.0, sx * ax, yw + 70.0, dt - 50.0), "bldc_emerald"))

        # Azimuth stage (local origin on the turntable axis at deck level)
        az: List[Part] = [
            Part(f"{side}_Azimuth_Platter", cyl(50.0, 12.0, 0, 0, dt + 10.0 - dt), "brass_gear"),
            Part(f"{side}_Mast_Post", cyl(16.0, z_axis - 22.0 - dt, 0, 90.0, 22.0), "titanium_frame"),
            Part(f"{side}_Post_Foot", box(40.0, 110.0, 12.0, 0, 45.0, 16.0), "titanium_frame"),
            Part(f"{side}_Yoke_Crossbar", box(w + 70.0, 24.0, 24.0, 0, 90.0, z_axis - dt), "titanium_frame"),
        ]
        for k, xx in enumerate((-(0.5 * w + 22.0), 0.5 * w + 22.0)):
            az.append(Part(f"{side}_Yoke_Arm_{k + 1}", box(16.0, 92.0, 24.0, xx, 46.0, z_axis - dt), "titanium_frame"))
            az.append(Part(f"{side}_Elevation_Bearing_{k + 1}", x_cyl(14.0, 20.0, xx, 0.0, z_axis - dt), "gearbox_gunmetal"))
        az.append(Part(f"{side}_Elevation_Motor", x_cyl(18.0, 50.0, 0.5 * w + 55.0, 0.0, z_axis - dt), "bldc_emerald"))

        # Elevation stage (local origin on the elevation axis): 2 panels stacked, cells on the -Y face
        el: List[Part] = []
        ph = 0.5 * h - 5.0
        for k, tag in enumerate(("Upper", "Lower")):
            zc = (1 if k == 0 else -1) * (0.5 * ph + 5.0)
            el.append(Part(f"{side}_Panel_{tag}_Substrate", box(w, 16.0, ph, 0, 8.0, zc), "solar_substrate"))
            el.append(Part(f"{side}_Panel_{tag}_GaInP_Cells", box(w - 30.0, 1.5, ph - 30.0, 0, -0.75, zc), "solar_cell"))
            for j in range(5):
                xx = -0.5 * (w - 30.0) + (j + 0.5) * (w - 30.0) / 5
                el.append(Part(f"{side}_Panel_{tag}_Busbar_{j + 1}", box(3.0, 1.0, ph - 30.0, xx, -2.0, zc), "solar_busbar"))
        for k, xx in enumerate((-(0.5 * w + 6.0), 0.5 * w + 6.0)):
            el.append(Part(f"{side}_Wing_Side_Frame_{k + 1}", box(12.0, 20.0, h, xx, 8.0, 0.0), "titanium_frame"))
            el.append(Part(f"{side}_Elevation_Pin_{k + 1}", x_cyl(8.0, 28.0, xx + (8.0 if xx > 0 else -8.0), 0.0, 0.0), "steel_shaft"))
        el.append(Part(f"{side}_Wing_Back_Spar", box(w, 14.0, 30.0, 0, 23.0, 0.0), "titanium_frame"))
        out[side] = {"az": az, "elev": el, "az_origin": (sx * ax, yw, dt), "elev_origin": (0.0, 0.0, z_axis - dt)}
    return out


def power_world_parts(pw: Dict[str, Any]) -> List[Part]:
    """All power-system parts placed in world coordinates at the default (home) pose."""
    parts = list(pw["static"])
    for side, _ in WING_SIDES:
        w = pw[side]
        o = bd.Vector(*w["az_origin"])
        e = o + bd.Vector(*w["elev_origin"])
        parts += [Part(q.name, bd.Pos(o.X, o.Y, o.Z) * q.shape, q.material) for q in w["az"]]
        parts += [Part(q.name, bd.Pos(e.X, e.Y, e.Z) * q.shape, q.material) for q in w["elev"]]
    return parts


# ==============================================================================
# EXPORT
# ==============================================================================
def part_to_trimesh(part: Part, tol: float = 0.08, ang: float = 0.25) -> trimesh.Trimesh:
    if any(k in part.name for k in COARSE_MESH + ("Transport_Helix_Flight", "Bristle_Tuft", "Brush_Torsion_Spring")):
        tol, ang = 0.25, 0.6
    verts, tris = part.shape.tessellate(tol, ang)
    v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=v, faces=np.array(tris, dtype=np.int64), process=False)
    key = colour_class(part.name)
    rgba, metallic, rough = _COLOUR[key]
    mesh.visual = TextureVisuals(material=PBRMaterial(
        name=f"fn_{key}",
        baseColorFactor=[_srgb_to_linear(c) for c in rgba[:3]] + [rgba[3] / 255.0],
        metallicFactor=metallic,
        roughnessFactor=rough,
        doubleSided=True,
    ))
    return mesh


def export_step(path: Path, groups: Dict[str, List[Part]], offsets: Optional[Dict[str, bd.Location]] = None) -> None:
    """Assembly-structured, colour-labelled STEP (mm)."""
    children = []
    for gname, parts in groups.items():
        loc = (offsets or {}).get(gname)
        kids = []
        for p in parts:
            s = p.shape.moved(loc if loc is not None else bd.Location())
            s.label = p.name
            s.color = _color(p.name)
            kids.append(s)
        comp = bd.Compound(children=kids)
        comp.label = gname
        children.append(comp)
    top = bd.Compound(children=children)
    top.label = path.stem
    bd.export_step(top, str(path))


def export_glb_stl(
    path_stem: Path,
    groups: Dict[str, List[Part]],
    node_translations_mm: Optional[Dict[str, Tuple[float, float, float]]] = None,
    named_nodes: bool = True,
    parents: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Write <stem>.glb (Y-up metres, one node per group, optional parent hierarchy) and <stem>.stl (mm)."""
    scene = trimesh.Scene()
    base = scene.graph.base_frame
    stl_meshes = []
    parents = parents or {}
    tr = node_translations_mm or {}

    def world_offset(g: str) -> np.ndarray:
        off = np.array(tr.get(g, (0.0, 0.0, 0.0)), dtype=float)
        while g in parents:
            g = parents[g]
            off = off + np.array(tr.get(g, (0.0, 0.0, 0.0)), dtype=float)
        return off

    for gname, parts in groups.items():
        t_mm = tuple(world_offset(gname)) if not named_nodes else tr.get(gname, (0.0, 0.0, 0.0))
        node_tf = BD_TO_GLTF @ trimesh.transformations.translation_matrix(t_mm) @ np.linalg.inv(BD_TO_GLTF)
        parent = gname if named_nodes else base
        if named_nodes:
            scene.graph.update(frame_from=parents.get(gname, base), frame_to=gname, matrix=node_tf)
        for p in parts:
            m = part_to_trimesh(p)
            world = m.copy()
            world.apply_translation(world_offset(gname))
            stl_meshes.append(world)
            m.apply_transform(BD_TO_GLTF)
            if not named_nodes:
                m.apply_transform(node_tf)
            node = f"{gname}__{p.name}" if named_nodes else p.name
            scene.add_geometry(m, node_name=node, geom_name=f"{node}_geom", parent_node_name=parent)

    glb = scene.export(file_type="glb", include_normals=True)
    path_stem.with_suffix(".glb").write_bytes(glb)
    combined = trimesh.util.concatenate(stl_meshes)
    combined.export(str(path_stem.with_suffix(".stl")))
    return {
        "parts": int(sum(len(v) for v in groups.values())),
        "triangles": int(len(combined.faces)),
        "glb_kb": round(len(glb) / 1024.0, 1),
        "bbox_mm": [[round(float(c), 1) for c in row] for row in combined.bounds],
    }


def exploded_bit(crown: List[Part], lay: Layout) -> List[Part]:
    """Exploded view: sampling section shortened to its first 40 mm (break view) so the small parts read."""
    keep = 40.0
    z_s0 = lay.protrusion + lay.params["drill_bit"]["crown_height_m"] * 1e3
    cut = lay.sample_len - keep
    clip = bd.Pos(0, 0, z_s0 + keep / 2) * bd.Box(40, 40, keep)
    out = []
    for q in crown:
        shape = q.shape
        dz = 0.0
        if q.name in ("Sampling_Root_Shaft", "Sampling_Flight"):
            shape = shape & clip
        elif q.name in ("Bit_Sub_Shoulder", "Bit_Sub_Pin_Thread", "Joint_Coupling"):
            dz = -cut
        ox, oy, oz = BIT_EXPLODE[q.name]
        out.append(Part(q.name, bd.Pos(ox, oy, oz + dz) * shape, q.material))
    return out


def exploded_brush(lay: Layout, frame: List[Part], brush_local: List[Part], carousel: List[Part]) -> List[Part]:
    """Brushing station pulled apart, with a short piece of the sampling auger left in mesh for context."""
    names = {q[0] for q in build_brush_frame(lay.params, lay.design)}
    out: List[Part] = []
    for q in frame:
        if q.name in names:
            dx, dy, dz = explode_offset(q.name)
            out.append(Part(q.name, bd.Pos(dx, dy, dz) * q.shape, q.material))
    for q in brush_local:
        dx, dy, dz = explode_offset(q.name)
        out.append(Part(q.name, bd.Pos(dx, lay.brush_y + dy, lay.brush_z + dz) * q.shape, q.material))
    ox, oy, oz = carousel_offset(lay)
    for q in carousel:
        dx, dy, dz = explode_offset(q.name)
        out.append(Part(q.name, bd.Pos(ox + dx, oy + dy, oz + dz) * q.shape, q.material))
    z_s0 = lay.protrusion + lay.params["drill_bit"]["crown_height_m"] * 1e3
    tip = lay.brush_z - z_s0 - 0.5 * lay.sample_len
    clip = bd.Pos(0, 0, lay.brush_z) * bd.Box(30, 30, 70)
    for q in build_auger(lay, tip):
        if q.name in ("Sampling_Root_Shaft", "Sampling_Flight"):
            out.append(Part(f"Mesh_Context_{q.name}", q.shape & clip, q.material))
    return out


def build_all(lay: Layout) -> Dict[str, Dict[str, List[Part]]]:
    zc = lay.zc_home
    head = build_drill_head(lay, zc)
    auger = build_auger(lay, lay.home_tip)
    return {
        "frame": build_frame(lay),
        "lander": build_lander_parts(lay),
        "power": build_power(lay),
        "carriage": head["carriage"],
        "spindle": head["spindle"],
        "hammer": head["hammer"],
        "cam": head["cam"],
        "auger": auger,
        "brush_local": build_brush_local(lay),
        "carousel": build_carousel_local(lay),
        "crown": [p for p in build_auger(lay, 0.0) if p.name in CROWN_PART_NAMES],
    }


def main() -> None:
    EXPORTS_CAD_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_CAD_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Module 3: Generating build123d B-rep CAD assemblies ===")
    params = load_system_params()
    design = derive_all(params)
    lay = load_layout(params, design)
    g = build_all(lay)
    pw = g["power"]
    brush_pos = (0.0, lay.brush_y, lay.brush_z)
    brush_loc = bd.Location(brush_pos)

    assemblies = {
        "auger_bit": {"Auger": g["auger"]},
        "drill_bit_crown": {"Bit_Crown": g["crown"]},
        "drill_bit_exploded": {"Bit_Exploded": exploded_bit(g["crown"], lay)},
        "drill_head": {"Carriage": g["carriage"], "Spindle": g["spindle"], "Hammer_Cam": g["hammer"], "Cam_Ring": g["cam"]},
        "drill_head_section": {"Head_Section": [Part(*t) for t in head_section(
            [(q.name, q.shape, q.material) for q in g["carriage"] + g["spindle"] + g["hammer"] + g["cam"]], lay.zc_home)]},
        "mast_and_brush": {"Frame": g["lander"] + g["frame"], "Brush_Gear": g["brush_local"], "Carousel": g["carousel"]},
        "mast_stowed": {"Stowed": g["lander"] + stowed_parts(lay, g["frame"], g["carriage"] + g["spindle"] + g["hammer"] + g["cam"]
                                                            + g["auger"], g["brush_local"], g["carousel"])},
        "brush_station_exploded": {"Brush_Exploded": exploded_brush(lay, g["frame"], g["brush_local"], g["carousel"])},
        "power_system": {"Solar_Power": power_world_parts(pw)},
    }
    car_pos = carousel_offset(lay)
    offsets_mm = {"Brush_Gear": brush_pos, "Carousel": car_pos}

    stats = {}
    for name, groups in assemblies.items():
        stats[name] = export_glb_stl(EXPORTS_CAD_DIR / name, groups, offsets_mm, named_nodes=False)
        export_step(EXPORTS_CAD_DIR / f"{name}.step", groups, {"Brush_Gear": brush_loc, "Carousel": bd.Location(car_pos)})

    rig_groups = {
        "Frame_Node": g["lander"] + g["frame"],
        "Solar_Node": pw["static"],
        "Carriage_Node": g["carriage"],
        "Auger_Node": g["auger"] + g["spindle"],
        "Hammer_Cam_Node": g["hammer"],
        "Cam_Ring_Node": g["cam"],
        "Brush_Gear_Node": g["brush_local"],
        "Carousel_Node": g["carousel"],
    }
    node_tr: Dict[str, Tuple[float, float, float]] = {"Brush_Gear_Node": brush_pos, "Carousel_Node": car_pos}
    parents: Dict[str, str] = {}
    step_offsets: Dict[str, bd.Location] = {"Brush_Gear_Node": brush_loc, "Carousel_Node": bd.Location(car_pos)}
    for side, _ in WING_SIDES:
        az_n, el_n = f"Solar_Wing_{side}_Az_Node", f"Solar_Wing_{side}_Elev_Node"
        rig_groups[az_n] = pw[side]["az"]
        rig_groups[el_n] = pw[side]["elev"]
        node_tr[az_n] = pw[side]["az_origin"]
        node_tr[el_n] = pw[side]["elev_origin"]
        parents[az_n] = "Solar_Node"
        parents[el_n] = az_n
        o, e = pw[side]["az_origin"], pw[side]["elev_origin"]
        step_offsets[az_n] = bd.Location(o)
        step_offsets[el_n] = bd.Location((o[0] + e[0], o[1] + e[1], o[2] + e[2]))
    stats["full_drill_rig"] = export_glb_stl(
        EXPORTS_CAD_DIR / "full_drill_rig", rig_groups, node_tr, named_nodes=True, parents=parents
    )
    export_step(EXPORTS_CAD_DIR / "full_drill_rig.step", rig_groups, step_offsets)

    for glb in EXPORTS_CAD_DIR.glob("*.glb"):
        shutil.copy2(glb, DOCS_CAD_DIR / glb.name)

    manifest = {
        "units": {"glb": "m (Y-up)", "step_stl": "mm (Z-up)"},
        "layout": {
            "auger_od_mm": lay.od,
            "auger_tube_od_mm": lay.od_tube,
            "auger_tube_id_mm": lay.id_tube,
            "sampling_flute_depth_mm": lay.flute_depth,
            "sampling_section_length_mm": lay.sample_len,
            "sampling_pitch_mm": lay.sample_pitch,
            "transport_pitch_mm": lay.upper_pitch,
            "auger_length_mm": lay.auger_len,
            "mast_top_mm": lay.mast_top,
            "mast_section_mm": design["mast"]["section_mm"],
            "mast_material": design["mast"]["label"],
            "deck_size_mm": [design["mast"]["geometry_m"]["deck_width"] * 1e3, design["mast"]["geometry_m"]["deck_length"] * 1e3],
            "deck_top_mm": lay.deck_top,
            "hinge_yz_mm": [design["mast"]["geometry_m"]["y_hinge"] * 1e3, design["mast"]["geometry_m"]["z_hinge"] * 1e3],
            "stow_carriage_drop_mm": design["mast"]["geometry_m"]["carriage_drop"] * 1e3,
            "wing_size_mm": [lay.wing_w, lay.wing_h],
            "wing_separation_mm": lay.wing_sep,
            "wing_station_y_mm": lay.wing_y,
            "sampling_root_od_mm": lay.root_od,
            "transport_helix_deg": design["auger"]["transport_helix_deg"],
            "rotary_gear_train": "8:1 planetary x 24T/12T spur = 16:1",
            "leadscrew_y_mm": lay.leadscrew_y,
            "head_stack_mm": design["head"]["stack_mm"],
            "brush_axle_offset_mm": -lay.brush_y,
            "brush_tip_radius_mm": design["brush"]["mesh"]["r_tip_m"] * 1e3,
            "guide_bushing_bore_mm": design["string"]["bushing"]["bore_m"] * 1e3,
        },
        "kinematics": {
            "home_tip_m": lay.home_tip / 1000.0,
            "brush_axle_m": [0.0, lay.brush_z / 1000.0, -lay.brush_y / 1000.0],
            "brush_radius_m": lay.brush_r / 1000.0,          # worm-wheel pitch radius (src/design_brush.py)
            "brush_tufts": design["brush"]["mesh"]["tufts"],
            "transport_pitch_m": lay.upper_pitch / 1000.0,
            "sampling_pitch_m": lay.sample_pitch / 1000.0,
            "sampling_zone_from_tip_m": [(lay.protrusion + params["drill_bit"]["crown_height_m"] * 1e3) / 1000.0,
                                         (lay.protrusion + params["drill_bit"]["crown_height_m"] * 1e3 + lay.sample_len) / 1000.0],
            "bit": {k: design["bit"][k] for k in ("edge_radius_m", "wedge_deg", "index_deg", "slot_width_m")},
            "wing_nodes": [f"Solar_Wing_{s}_Az_Node" for s, _ in WING_SIDES],
            "cam_lift_m": params["actuators"]["percussive"]["cam_lift_m"],
            "blow_hz": design["hammer"]["blow_frequency_hz"],
            "cam_rps": params["actuators"]["percussive"]["cam_speed_rps"],
            # Sample carousel (glTF metres): turn Carousel_Node by +s x pitch about +Y to put cup s under the port
            "carousel": {
                "axis_m": [0.0, 0.0, design["brush"]["carousel"]["axis_offset_m"]],
                "pitch_radius_m": design["brush"]["carousel"]["pitch_radius_m"],
                "stations": design["brush"]["carousel"]["stations"],
                "cups": design["brush"]["carousel"]["cups"],
                "port_deg": 90.0,
                "cup_id_m": design["brush"]["cup"]["id_m"],
                "cup_floor_m": design["brush"]["cup"]["z0_m"] + design["brush"]["cup"]["wall_m"],
                "cup_rim_m": design["brush"]["cup"]["rim_m"],
                "index_time_s": design["brush"]["carousel"]["index_time_s"],
            },
            "chute": {
                "inlet_m": [0.0, design["brush"]["chute"]["z_in_m"], design["brush"]["chute"]["y_in_m"]],
                "outlet_m": [0.0, design["brush"]["chute"]["z_out_m"], design["brush"]["cup"]["y_m"]],
                "plate_top_m": params["cad_layout"]["foot_top_m"],
            },
        },
        "colour_code": [{"key": k, "label": lab, "function": fn, "hex": "#%02x%02x%02x" % rgba[:3]}
                        for k, lab, fn, rgba, _, _, _ in COLOUR_CODE],
        "nodes": {k: [p.name for p in v] for k, v in rig_groups.items()},
        "assemblies": stats,
    }
    (EXPORTS_DATA_DIR / "cad_manifest.json").write_text(json.dumps(manifest, indent=2))

    for name, s in stats.items():
        print(f"  {name:16s}: {s['parts']:4d} parts | {s['triangles']:7d} tris | GLB {s['glb_kb']:7.1f} KB")
    print("  STEP (mm, labelled assemblies), STL (mm) and GLB (m) written; GLBs synced to docs_src/assets/cad/.")


if __name__ == "__main__":
    main()
