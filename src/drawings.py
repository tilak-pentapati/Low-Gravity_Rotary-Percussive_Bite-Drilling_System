"""
Module 7: Dimensioned 2D engineering drawings (A3, third-angle projection) from the build123d B-rep.

  * Views are hidden-line-removed projections of the same solids that src/cad_generator.py exports
    (build123d `project_to_viewport`); section views cut the solids on the axial plane and hatch the cut faces.
  * Every dimension is either a config / derived value or measured from the solid it annotates, so the
    drawings change when the design does.
  * Sheets are laid out in paper millimetres with matplotlib and written as SVG + PDF to exports/drawings/.

Sheets:  DRL-001 drill bit crown and bit sub     DRL-002 enclosed rotary-percussive drill head
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import build123d as bd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from src.cad_bit import build_bit  # noqa: E402
from src.cad_generator import build_drill_head, load_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "exports" / "drawings"
A3 = (420.0, 297.0)
AUTHOR = "Jay Tilak P."
DRAWN_DATE = "2022-09-07"
TXT = 2.5 / 0.3528  # 2.5 mm lettering in points
LW_VIS, LW_HID, LW_THIN = 0.7, 0.35, 0.25
HATCH = {"tungsten_carbide": "xxxx", "steel_shaft": "////", "flute_hardened": "////", "copper_winding": "....",
         "brass_gear": "\\\\\\\\", "gearbox_gunmetal": "////", "titanium": "\\\\\\\\"}


# ------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------
def _polyline(edge: bd.Edge) -> np.ndarray:
    if edge.geom_type == bd.GeomType.LINE:
        pts = [edge.position_at(0), edge.position_at(1)]
    else:
        n = int(np.clip(edge.length / 0.4, 8, 160))
        pts = edge.positions(list(np.linspace(0, 1, n)))
    return np.array([[p.X, p.Y] for p in pts])


def project(shapes: Sequence[bd.Shape], eye: Tuple[float, float, float], up: Tuple[float, float, float],
            target: Tuple[float, float, float]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Hidden-line projection: (visible, hidden) 2D polylines in view coordinates (mm, origin at target)."""
    vis, hid = bd.Compound(children=list(shapes)).project_to_viewport(eye, up, target)
    return [_polyline(e) for e in vis], [_polyline(e) for e in hid]


def half_space(y_keep_min: float = 0.0) -> bd.Part:
    return bd.Pos(0, 1000 + y_keep_min, 0) * bd.Box(4000, 2000, 6000)


def section_parts(parts: Iterable[Tuple[str, bd.Shape, str]]):
    """Cut every part on the plane y = 0 and keep y >= 0; returns (cut solids, cut faces with material)."""
    kept, faces = [], []
    box = half_space()
    for name, shape, mat in parts:
        bb = shape.bounding_box()
        if bb.max.Y <= 1e-6:
            continue
        if bb.min.Y >= -1e-6:
            kept.append(shape)
            continue
        try:
            cut = shape & box
        except Exception:
            continue
        if cut is None or not cut.solids():
            continue
        kept.append(cut)
        for f in cut.faces():
            if abs(f.center().Y) < 1e-4 and abs(abs(f.normal_at().Y) - 1) < 1e-6:
                faces.append((f, mat))
    return kept, faces


def _wire_xy(w: bd.Wire, x_of, y_of) -> np.ndarray:
    n = int(np.clip(w.length / 0.3, 12, 400))
    pts = np.array([[x_of(p), y_of(p)] for p in w.positions(list(np.linspace(0, 1, n)))])
    area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    return pts, area


def face_path(f: bd.Face, x_of, y_of) -> MPath:
    """Matplotlib path of a planar face with holes (outer CCW, inner CW) for hatching."""
    verts, codes = [], []
    outer, a = _wire_xy(f.outer_wire(), x_of, y_of)
    loops = [outer if a > 0 else outer[::-1]]
    for w in f.inner_wires():
        p, a = _wire_xy(w, x_of, y_of)
        loops.append(p if a < 0 else p[::-1])
    for p in loops:
        verts += list(p) + [p[0]]
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(p) - 1) + [MPath.CLOSEPOLY]
    return MPath(np.array(verts), codes)


# ------------------------------------------------------------------------------
# Sheet / annotation helpers (all in paper mm)
# ------------------------------------------------------------------------------
class Sheet:
    def __init__(self, title: str, number: str, scale_note: str, notes: List[str], material: str):
        self.fig = plt.figure(figsize=(A3[0] / 25.4, A3[1] / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, A3[0])
        self.ax.set_ylim(0, A3[1])
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.title, self.number = title, number
        self._frame(scale_note, notes, material)

    def _frame(self, scale_note: str, notes: List[str], material: str) -> None:
        ax = self.ax
        ax.add_patch(Rectangle((10, 10), A3[0] - 20, A3[1] - 20, fill=False, lw=1.0))
        # Title block (bottom right, 180 x 44)
        x0, y0, w, h = A3[0] - 190, 10, 180, 44
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=0.8))
        for yy in (y0 + 14, y0 + 26):
            ax.plot([x0, x0 + w], [yy, yy], lw=0.4, color="k")
        ax.plot([x0 + 110, x0 + 110], [y0, y0 + 26], lw=0.4, color="k")
        ax.plot([x0 + 145, x0 + 145], [y0, y0 + 14], lw=0.4, color="k")
        ax.text(x0 + 3, y0 + 37, self.title, fontsize=TXT * 1.5, weight="bold", va="center")
        ax.text(x0 + 3, y0 + 30, "Autonomous low-gravity rotary-percussive bite-drilling system", fontsize=TXT * 0.85,
                va="center", color="#333")
        ax.text(x0 + 3, y0 + 20, f"Material: {material}", fontsize=TXT * 0.8, va="center")
        ax.text(x0 + 3, y0 + 7, f"Drawn: {AUTHOR}    Date: {DRAWN_DATE}", fontsize=TXT * 0.8,
                va="center")
        ax.text(x0 + 113, y0 + 20, f"Scale: {scale_note}\nUnits: mm   Sheet: A3", fontsize=TXT * 0.8, va="center")
        ax.text(x0 + 113, y0 + 7, "DWG NO.", fontsize=TXT * 0.6, va="center", color="#555")
        ax.text(x0 + 127, y0 + 7, self.number, fontsize=TXT * 1.1, weight="bold", va="center")
        ax.text(x0 + 148, y0 + 7, "REV A", fontsize=TXT, va="center")
        self._third_angle(x0 + 160, y0 + 20)
        # General notes (above the title block)
        ny = y0 + h + 6
        for i, n in enumerate(reversed(notes)):
            ax.text(x0, ny + 4.2 * i, n, fontsize=TXT * 0.8, va="bottom")
        ax.text(x0, ny + 4.2 * len(notes), "NOTES", fontsize=TXT * 0.9, weight="bold", va="bottom")

    def _third_angle(self, x: float, y: float) -> None:
        ax = self.ax
        ax.plot([x - 9, x - 1, x - 1, x - 9, x - 9], [y - 2.5, y - 4, y + 4, y + 2.5, y - 2.5], lw=0.5, color="k")
        for r in (4.0, 2.0):
            t = np.linspace(0, 2 * np.pi, 60)
            ax.plot(x + 7 + r * np.cos(t), y + r * np.sin(t), lw=0.5, color="k")
        ax.plot([x - 11, x + 13], [y, y], lw=0.3, color="k", dashes=(6, 1.5, 1, 1.5))

    def view(self, vis, hid, origin, s, rot: bool = False, hidden: bool = True) -> None:
        for pl, lw, style in [(vis, LW_VIS, "-")] + ([(hid, LW_HID, (0, (3, 1.5)))] if hidden else []):
            for p in pl:
                q = p[:, ::-1] * [1, -1] if rot else p
                self.ax.plot(origin[0] + s * q[:, 0], origin[1] + s * q[:, 1], color="k", lw=lw, linestyle=style,
                             solid_capstyle="round")

    def hatch(self, faces, origin, s, x_of, y_of) -> None:
        for f, mat in faces:
            path = face_path(f, lambda p: origin[0] + s * x_of(p), lambda p: origin[1] + s * y_of(p))
            self.ax.add_patch(PathPatch(path, facecolor="none", edgecolor="#444", lw=0, hatch=HATCH.get(mat, "////")))

    def centerline(self, p1, p2) -> None:
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="k", lw=LW_THIN, dashes=(10, 2, 2, 2))

    def label(self, x, y, text, size=1.0, **kw) -> None:
        self.ax.text(x, y, text, fontsize=TXT * size, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)

    def dim(self, p1, p2, offset: float, text: str, horizontal: bool) -> None:
        """Linear dimension between paper points p1, p2; dimension line `offset` mm away (sign = side)."""
        ax = self.ax
        if horizontal:
            yd = max(p1[1], p2[1]) + offset if offset > 0 else min(p1[1], p2[1]) + offset
            for p in (p1, p2):
                ax.plot([p[0], p[0]], [p[1] + np.sign(offset) * 1.0, yd + np.sign(offset) * 1.5], lw=LW_THIN, color="k")
            a, b = (p1[0], yd), (p2[0], yd)
            tpos, rot, va = ((a[0] + b[0]) / 2, yd + 1.0), 0, "bottom"
        else:
            xd = max(p1[0], p2[0]) + offset if offset > 0 else min(p1[0], p2[0]) + offset
            for p in (p1, p2):
                ax.plot([p[0] + np.sign(offset) * 1.0, xd + np.sign(offset) * 1.5], [p[1], p[1]], lw=LW_THIN, color="k")
            a, b = (xd, p1[1]), (xd, p2[1])
            tpos, rot, va = (xd - 1.0, (a[1] + b[1]) / 2), 90, "bottom"
        ax.annotate("", xy=a, xytext=b, arrowprops=dict(arrowstyle="<|-|>,head_length=0.5,head_width=0.18", lw=LW_THIN,
                                                        color="k", shrinkA=0, shrinkB=0))
        ax.text(*tpos, text, fontsize=TXT, ha="center", va=va, rotation=rot,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))

    def leader(self, xy, xytext, text: str) -> None:
        self.ax.annotate(text, xy=xy, xytext=xytext, fontsize=TXT * 0.9, ha="left" if xytext[0] >= xy[0] else "right",
                         va="center", arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.15", lw=LW_THIN,
                                                      color="k", shrinkA=1, shrinkB=0))

    def save(self, stem: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "pdf"):
            self.fig.savefig(OUT_DIR / f"{stem}.{ext}", format=ext)
        plt.close(self.fig)


def bbox_of(parts, name):
    return next(s for n, s, _ in parts if n == name).bounding_box()


# ------------------------------------------------------------------------------
# DRL-001  Drill bit
# ------------------------------------------------------------------------------
def drawing_bit(lay) -> dict:
    p, d = lay.params, lay.design
    b, a, db, da = p["drill_bit"], p["auger_geometry"], d["bit"], d["auger"]
    parts = build_bit(p, d, 0.0)
    crown_names = {"WC_Insert_1", "WC_Insert_2", "Bit_Crown_Body", "Gauge_Reamer_Pad_1", "Gauge_Reamer_Pad_2"}
    crown = [t for t in parts if t[0] in crown_names]
    z_face = a["cutter_protrusion_m"] * 1e3
    z_top = z_face + b["crown_height_m"] * 1e3
    gauge_d = a["gauge_diameter_m"] * 1e3
    body_d = 2 * db["body_r_m"] * 1e3
    root_d = da["root_od_mm"]
    clip = bd.Pos(0, 0, z_top + 5) * bd.Box(60, 60, 10)  # short stub of the root, then a break line
    stub = [(n, s & clip, m) for n, s, m in parts if n in ("Sampling_Root_Shaft", "Thermocouple_Probe")]

    notes = [
        f"1. Inserts: {b['insert_grade']}, {b['insert_thickness_m'] * 1e3:.1f} x {b['insert_height_m'] * 1e3:.0f} mm chisel, "
        f"silver-brazed, {b['braze_gap_m'] * 1e6:.0f} um joint.",
        f"2. Rake {b['rake_angle_deg']:.0f} deg, clearance {b['clearance_angle_deg']:.0f} deg, edge hone "
        f"R{db['edge_radius_m'] * 1e3:.1f} (derived: Hertz contact on a basalt clast).",
        f"3. Gauge pads WC; crown body 1 mm under gauge; {db['slot_width_m'] * 1e3:.0f} mm junk slots ahead of each insert.",
        f"4. Joint {b['joint_thread']} RH, {b['joint_engagement_m'] * 1e3:.0f} mm engagement; impact passes through the "
        f"{b['joint_shoulder_od_m'] * 1e3:.0f} mm shoulder.",
        "5. Type-T thermocouple pocket ends 0.5 mm above the insert tops.",
        "6. General tolerances ISO 2768-mK. Break sharp edges 0.2 unless shown.",
    ]
    sh = Sheet("DRILL BIT CROWN & BIT SUB", "DRL-001", "3:1  (bit sub 1:1)", notes,
               f"Body {b['body_steel']}")
    s = 3.0

    # View 1 - front (from -Y), crown + root stub; hidden lines dashed
    lz = 0.0
    vis, hid = project([t[1] for t in crown + stub], (0, -1000, lz), (0, 0, 1), (0, 0, lz))
    fo = (80.0, 120.0)  # paper position of (x=0, z=0)
    sh.view(vis, hid, fo, s)
    sh.centerline((fo[0], fo[1] - 6), (fo[0], fo[1] + s * (z_top + 11)))
    zb = z_top + 8.5  # break line
    xs = np.linspace(-root_d / 2 - 2, root_d / 2 + 2, 30)
    sh.ax.plot(fo[0] + s * xs, fo[1] + s * (zb + 0.6 * np.sin(xs * 1.4)), lw=LW_THIN, color="k")
    sh.label(fo[0], fo[1] + s * (z_top + 15), "FRONT VIEW", 1.0, weight="bold")
    sh.dim((fo[0] - s * gauge_d / 2, fo[1]), (fo[0] + s * gauge_d / 2, fo[1]), -10, f"Ø{gauge_d:.0f} gauge", True)
    sh.dim((fo[0] - s * body_d / 2, fo[1] + s * z_top), (fo[0] + s * body_d / 2, fo[1] + s * z_top), 9,
           f"Ø{body_d:.0f} body", True)
    sh.dim((fo[0] + s * gauge_d / 2, fo[1]), (fo[0] + s * gauge_d / 2, fo[1] + s * z_face), 8,
           f"{z_face:.0f}", False)
    sh.dim((fo[0] + s * body_d / 2, fo[1] + s * z_face), (fo[0] + s * body_d / 2, fo[1] + s * z_top), 22,
           f"{z_top - z_face:.0f} crown", False)
    sh.leader((fo[0] + s * root_d / 2, fo[1] + s * (z_top + 5)), (fo[0] + 30, fo[1] + s * (z_top + 9)),
              f"Ø{root_d:.0f} sampling root")
    sh.leader((fo[0] - s * 8.5, fo[1] + s * 5), (fo[0] - 45, fo[1] + 40), "WC insert (x2)")
    pad = bbox_of(parts, "Gauge_Reamer_Pad_2")
    sh.leader((fo[0] - s * 6.5, fo[1] + s * 0.5 * (pad.min.Z + pad.max.Z)), (fo[0] - 45, fo[1] + 62), "Gauge pad (x2)")

    # View 2 - bottom (third angle: below the front view), cutting face
    vis, hid = project([t[1] for t in crown], (0, 0, -1000), (0, -1, 0), (0, 0, 0))
    bo = (fo[0], 60.0)
    sh.view(vis, [], bo, s, hidden=False)
    sh.centerline((bo[0] - s * 13, bo[1]), (bo[0] + s * 13, bo[1]))
    sh.centerline((bo[0], bo[1] - s * 13), (bo[0], bo[1] + s * 13))
    sh.label(bo[0], bo[1] - s * 13 - 4, "BOTTOM VIEW (cutting face)", 1.0, weight="bold")
    sh.dim((bo[0] - s * db['slot_width_m'] * 1e3 / 2, bo[1] + s * 9), (bo[0] + s * db['slot_width_m'] * 1e3 / 2,
                                                                     bo[1] + s * 9), 4,
           f"{db['slot_width_m'] * 1e3:.0f} slot", True)
    t_ins = b["insert_thickness_m"] * 1e3
    sh.leader((bo[0] + s * 6, bo[1]), (bo[0] + 48, bo[1] - 14), f"Insert {t_ins:.1f} thk, index {db['index_deg']:.1f}° / blow")

    # View 3 - section A-A on the insert plane (y = 0), crown + stub
    cut, faces = section_parts(crown + stub)
    vis, _ = project(cut, (0, -1000, 0), (0, 0, 1), (0, 0, 0))
    so = (205.0, 120.0)
    sh.view(vis, [], so, s, hidden=False)
    sh.hatch(faces, so, s, lambda q: q.X, lambda q: q.Z)
    sh.centerline((so[0], so[1] - 6), (so[0], so[1] + s * (z_top + 11)))
    sh.ax.plot(so[0] + s * xs, so[1] + s * (zb + 0.6 * np.sin(xs * 1.4)), lw=LW_THIN, color="k")
    sh.label(so[0], so[1] + s * (z_top + 15), "SECTION A-A", 1.0, weight="bold")
    pocket = z_face + b["insert_height_m"] * 1e3 - z_face + b["thermocouple_standoff_m"] * 1e3
    sh.dim((so[0], so[1]), (so[0], so[1] + s * pocket), -40, f"{pocket:.1f}", False)
    sh.leader((so[0] + s * a['wiring_bore_m'] * 1e3 / 2, so[1] + s * (z_top + 5)), (so[0] + 30, so[1] + s * (z_top + 9)),
              f"Ø{a['wiring_bore_m'] * 1e3:.0f} wiring bore")
    sh.leader((so[0] + s * 1.5, so[1] + s * 6), (so[0] + 50, so[1] + 10), "WC insert, brazed")
    sh.leader((so[0] + s * 0.4, so[1] + s * 20), (so[0] + 50, so[1] + 50), "Thermocouple probe")
    sh.leader((so[0] - s * 0.5, so[1] + s * pocket), (so[0] - 50, so[1] + s * pocket + 14), "TC pocket end")
    # section-line marks on the bottom view
    for xx in (-1, 1):
        sh.ax.plot([bo[0] + xx * s * 13, bo[0] + xx * s * 15.5], [bo[1], bo[1]], lw=1.2, color="k")
        sh.label(bo[0] + xx * s * 17, bo[1] + 3, "A", 1.1, weight="bold")

    # View 4 - complete bit sub, 1:1, axis horizontal (top band)
    sub = [t for t in parts if t[0] not in ("Thermocouple_Probe",)]
    vis, _ = project([t[1] for t in sub], (0, -1000, 0), (0, 0, 1), (0, 0, 0))
    vo = (60.0, 268.0)
    for pl in vis:
        sh.ax.plot(vo[0] + pl[:, 1], vo[1] - pl[:, 0], color="k", lw=LW_VIS * 0.8)
    L = bbox_of(parts, "Joint_Coupling").max.Z
    zs0, l_s = z_top, da["sampling_length_mm"]
    sh.centerline((vo[0] - 5, vo[1]), (vo[0] + L + 5, vo[1]))
    sh.label(vo[0] + L + 30, vo[1] - 14, "BIT SUB ASSEMBLY  1:1", 1.0, weight="bold")
    sh.dim((vo[0], vo[1] - gauge_d / 2), (vo[0] + L, vo[1] - gauge_d / 2), -9, f"{L:.0f} overall", True)
    sh.dim((vo[0] + zs0, vo[1] + 10), (vo[0] + zs0 + l_s, vo[1] + 10), 6,
           f"{l_s:.0f} sampling flutes (holds one {p['system']['bite_depth_m'] * 100:.0f} cm bite)"
           if "bite_depth_m" in p.get("system", {}) else f"{l_s:.0f} sampling flutes", True)
    pitch = da["sampling_pitch_mm"]
    x1 = vo[0] + zs0 + 1 + 3 * pitch
    sh.dim((x1, vo[1] + 10), (x1 + pitch, vo[1] + 10), -3.5, f"P{pitch:.0f}", True)
    sh.leader((vo[0] + L - 6, vo[1] + 6), (vo[0] + L + 14, vo[1] + 16), f"{b['joint_thread']} joint")
    sh.leader((vo[0] + 12, vo[1] - 9), (vo[0] - 2, vo[1] - 20), "Crown (detail views)")
    sh.save("DRL-001_drill_bit")
    return {"number": "DRL-001", "title": "Drill bit crown & bit sub", "overall_mm": round(L, 1)}


# ------------------------------------------------------------------------------
# DRL-002  Drill head
# ------------------------------------------------------------------------------
def drawing_head(lay) -> dict:
    p, d = lay.params, lay.design
    zc = lay.zc_home
    groups = build_drill_head(lay, zc)
    parts = [(q.name, q.shape, q.material) for v in groups.values() for q in v]
    shapes = [t[1] for t in parts]
    bb = bd.Compound(children=shapes).bounding_box()
    H, W = bb.max.Z - bb.min.Z, bb.max.X - bb.min.X
    st, plan = d["head"]["stack_mm"], d["head"]["plan_mm"]
    s = 0.5
    z0 = bb.min.Z
    notes = [
        f"1. Rotary: frameless BLDC -> 12/36/84 planetary (3 planets) -> 12T/24T spur; spindle on SKF 7004 ACD pair.",
        f"2. Percussion: 2-lobe barrel cam lifts the {d['ledger']['hammer_mass_kg']['value']:.3f} kg hammer against a "
        f"{d['hammer']['spring']['wire_mm']:.0f} mm 17-7PH spring; {d['hammer']['impact_energy_j']:.2f} J at "
        f"{d['hammer']['blow_frequency_hz']:.0f} Hz.",
        f"3. Feed: Tr16x4 bronze nut, self-locking; WOB measured by the {d['head']['load_cell_n']:.0f} N load cell under the yoke.",
        "4. Housing Ti-6Al-4V; labyrinth + PTFE lip seals and dust bellows exclude regolith.",
        "5. Bearing lives and gear checks: see the design-derivation ledger. General tolerances ISO 2768-mK.",
    ]
    sh = Sheet("ENCLOSED ROTARY-PERCUSSIVE DRILL HEAD", "DRL-002", "1:2", notes, "Housing Ti-6Al-4V; gears 9310")
    look = (0, 0, z0)

    # Front view (from -Y)
    vis, _ = project(shapes, (0, -3000, z0), (0, 0, 1), look)
    fo = (80.0, 102.0)
    sh.view(vis, [], fo, s, hidden=False)
    sh.centerline((fo[0], fo[1] - 6), (fo[0], fo[1] + s * H + 6))
    sh.label(fo[0], fo[1] + s * H + 12, "FRONT VIEW", 1.0, weight="bold")
    sh.dim((fo[0] + s * bb.min.X, fo[1]), (fo[0] + s * bb.max.X, fo[1]), -12, f"{W:.0f}", True)
    sh.dim((fo[0] + s * bb.min.X, fo[1]), (fo[0] + s * bb.min.X, fo[1] + s * H), -8, f"{H:.0f}", False)
    rx = lay.rail_x
    sh.dim((fo[0] - s * rx, fo[1] + s * H), (fo[0] + s * rx, fo[1] + s * H), 6, f"{2 * rx:.0f} rail centres", True)

    # Right side view (from +X), third angle -> placed right of the front view
    vis, _ = project(shapes, (3000, 0, z0), (0, 0, 1), look)
    ro = (192.0, 102.0)
    sh.view(vis, [], ro, s, hidden=False)
    sh.centerline((ro[0], ro[1] - 6), (ro[0], ro[1] + s * H + 6))
    sh.label(ro[0] - 22, ro[1] + s * H + 12, "RIGHT SIDE VIEW", 1.0, weight="bold")
    ly = plan["leadscrew_y"]
    sh.centerline((ro[0] + s * ly, ro[1] + s * (H - 30)), (ro[0] + s * ly, ro[1] + s * H + 4))
    sh.dim((ro[0], ro[1] + s * H), (ro[0] + s * ly, ro[1] + s * H), 6, f"{ly:.0f}", True)
    sh.label(ro[0] + s * ly + 2, ro[1] + s * H - 22, "leadscrew\naxis", 0.8, ha="left")
    sh.dim((ro[0] + s * bb.min.Y, ro[1]), (ro[0] + s * bb.max.Y, ro[1]), -14, f"{bb.max.Y - bb.min.Y:.0f}", True)

    # Section A-A on the spindle axis (plane y = 0), placed far right
    cut, faces = section_parts(parts)
    vis, _ = project(cut, (0, -3000, z0), (0, 0, 1), look)
    so = (318.0, 102.0)
    sh.view(vis, [], so, s, hidden=False)
    sh.hatch(faces, so, s, lambda q: q.X, lambda q: q.Z - z0)
    sh.centerline((so[0], so[1] - 6), (so[0], so[1] + s * H + 6))
    sh.label(so[0], so[1] + s * H + 12, "SECTION A-A (spindle axis)", 1.0, weight="bold")
    callouts = [("Hammer_Mass", "Hammer"), ("Hammer_Compression_Spring", "Hammer spring"),
                ("Barrel_Cam_Track_2_Lobe", "Barrel cam"), ("Anvil", "Anvil / hex socket"),
                ("Spindle_Bearing_7004ACD_A_Rings", "Angular-contact pair"), ("Planetary_Ring_Gear_84T", "Planetary"),
                ("Rotary_BLDC_Stator", "Rotary BLDC"), ("WOB_Load_Cell_500N", "WOB load cell"),
                ("Slip_Ring_Stator", "Slip ring")]
    tx = so[0] + s * bb.max.X + 12
    ys = []
    for name, text in callouts:
        try:
            q = bbox_of(parts, name)
        except StopIteration:
            continue
        ys.append(((q.min.Z + q.max.Z) / 2, q, text))
    ys.sort(key=lambda t: t[0])
    for i, (zm, q, text) in enumerate(ys):
        x_pt = so[0] + s * min(max(q.max.X, 2.0), bb.max.X) * 0.9
        y_txt = so[1] + 4 + i * (s * H - 8) / max(len(ys) - 1, 1)
        sh.ax.annotate("", xy=(x_pt, so[1] + s * (zm - z0)), xytext=(tx - 2.6, y_txt),
                       arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.15", lw=LW_THIN, color="k"))
        sh.ax.add_patch(plt.Circle((tx, y_txt), 2.6, fill=False, lw=0.5))
        sh.label(tx, y_txt, str(i + 1), 0.85)
    # cutting plane A-A (y = 0) marked on the side view
    for zz in (-8.0, s * H + 1.0):
        sh.ax.plot([ro[0], ro[0]], [ro[1] + zz, ro[1] + zz + 6], lw=1.2, color="k")
        sh.label(ro[0] + 4, ro[1] + zz + 3, "A", 1.1, weight="bold")
    # item list (bottom left)
    for k, (zm, q, text) in enumerate(ys):
        cx, cy = 20 + 70 * (k // 3), 48 - 9 * (k % 3)
        sh.ax.add_patch(plt.Circle((cx, cy), 2.6, fill=False, lw=0.5))
        sh.label(cx, cy, str(k + 1), 0.85)
        sh.label(cx + 5, cy, text, 0.85, ha="left")
    sh.label(15, 58, "ITEMS (section A-A)", 0.9, ha="left", weight="bold")
    sh.save("DRL-002_drill_head")
    return {"number": "DRL-002", "title": "Enclosed rotary-percussive drill head", "envelope_mm": [round(W), round(H)]}


def main() -> None:
    print("=== Module 7: Dimensioned engineering drawings ===")
    lay = load_layout()
    index = {"bit": drawing_bit(lay), "head": drawing_head(lay)}
    (OUT_DIR / "drawings.json").write_text(json.dumps(index, indent=2))
    for k, v in index.items():
        print(f"  {v['number']}: {v['title']}")


if __name__ == "__main__":
    main()
