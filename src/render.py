"""
Module 6: Reproducible still renders of the CAD assemblies.

Reads the exported glTF scenes (exports/cad/*.glb), rasterises them with a small z-buffer renderer
(numba, no GPU or display needed) and writes PNG stills to exports/renders/:

  * smooth per-vertex normals within each B-rep face (the tessellation keeps faces separate, so
    edges between faces stay crisp), two-light Blinn-Phong with a hemispheric ambient term;
  * metallic / base colour from each part's glTF PBR material;
  * silhouette and crease lines from depth / normal / part discontinuities (technical-illustration look);
  * 2x supersampling, then a box filter down to the output size.

Every view (camera azimuth / elevation / framing) is listed in VIEWS, so `make renders` rebuilds identical images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from numba import njit
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAD_DIR = ROOT / "exports" / "cad"
OUT_DIR = ROOT / "exports" / "renders"

SS = 2  # supersampling factor

# name: (glb stem, azimuth deg, elevation deg, width px, height px, clip z mm, caption)
VIEWS: Dict[str, Tuple[str, float, float, int, int, str]] = {
    "rig_deployed": ("full_drill_rig", -38.0, 18.0, 1600, 1000, np.inf,
                     "Full rig, mast deployed: lander deck, side-mounted feed mast, drill head and 2-axis solar wings"),
    "rig_stowed": ("mast_stowed", -38.0, 24.0, 1200, 1200, np.inf,
                   "Launch configuration: mast folded aft over the deck on two hold-downs"),
    "mast_deployed": ("mast_and_brush", -32.0, 14.0, 1200, 1200, np.inf,
                      "Deployed mast on the lander deck: rails, leadscrew and brushing station"),
    "bit_closeup": ("drill_bit_crown", -30.0, -24.0, 1200, 1200, 62.0,
                    "Bit crown: two WC chisel inserts, gauge pads, junk slots and the sampling flutes"),
    "bit_exploded": ("drill_bit_exploded", -35.0, 14.0, 1200, 1200, np.inf,
                     "Exploded bit: inserts, crown body, gauge pads, thermocouple, sampling sub and M12x1 joint"),
    "head_section": ("drill_head_section", 35.0, 16.0, 1200, 1200, np.inf,
                     "Drill head quarter section: BLDC motors, planetary, cam-and-spring hammer, anvil and spindle"),
    "brush_exploded": ("brush_station_exploded", -50.0, 22.0, 1200, 1200, np.inf,
                       "Exploded brushing station: worm-wheel brush, spring arm, chute and sample cup"),
}


# ------------------------------------------------------------------------------
# Scene loading
# ------------------------------------------------------------------------------
def load_scene(path: Path):
    """World-space triangles in Z-up mm, per-vertex normals, per-triangle colour / metallic / part id."""
    sc = trimesh.load(path)
    V, N, F, col, met, pid = [], [], [], [], [], []
    off = 0
    for k, node in enumerate(sc.graph.nodes_geometry):
        T, g = sc.graph[node]
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        v = m.vertices[:, [0, 2, 1]] * np.array([1e3, -1e3, 1e3])  # Y-up m -> Z-up mm
        f = m.faces[:, [0, 2, 1]]  # the axis swap mirrors, so flip winding
        n = trimesh.Trimesh(v, f, process=False).vertex_normals
        mat = m.visual.material
        rgba = np.asarray(getattr(mat, "baseColorFactor", None) if getattr(mat, "baseColorFactor", None) is not None
                          else [180, 180, 180, 255], dtype=float)
        if rgba.max() > 1.0:
            rgba = rgba / 255.0
        # glTF base colours are linear RGB; the shading below works in display (sRGB) values
        rgba[:3] = np.where(rgba[:3] <= 0.0031308, 12.92 * rgba[:3], 1.055 * rgba[:3] ** (1 / 2.4) - 0.055)
        V.append(v); N.append(n); F.append(f + off)
        col.append(np.repeat(rgba[None, :3], len(f), 0))
        met.append(np.full(len(f), float(getattr(mat, "metallicFactor", 0.0) or 0.0)))
        pid.append(np.full(len(f), k))
        off += len(v)
    return (np.vstack(V), np.vstack(N), np.vstack(F).astype(np.int64), np.vstack(col), np.concatenate(met),
            np.concatenate(pid).astype(np.int64))


# ------------------------------------------------------------------------------
# Rasteriser
# ------------------------------------------------------------------------------
@njit(cache=True)
def _raster(sx, sy, sz, wz, zmax, F, vn, W, H, depth, nbuf, fid):
    for t in range(F.shape[0]):
        i0, i1, i2 = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = sx[i0], sy[i0], sz[i0]
        x1, y1, z1 = sx[i1], sy[i1], sz[i1]
        x2, y2, z2 = sx[i2], sy[i2], sz[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        for py in range(ymin, ymax + 1):
            fy = py + 0.5
            for px in range(xmin, xmax + 1):
                fx = px + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) / area
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z < depth[py, px] and w0 * wz[i0] + w1 * wz[i1] + w2 * wz[i2] <= zmax:
                    depth[py, px] = z
                    fid[py, px] = t
                    for c in range(3):
                        nbuf[py, px, c] = w0 * vn[i0, c] + w1 * vn[i1, c] + w2 * vn[i2, c]


def _camera(V: np.ndarray, az: float, el: float):
    """Orthographic-like long-lens camera: returns rotation (rows = right, up, back) and centre."""
    a, e = np.radians(az), np.radians(el)
    back = np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)])  # from target towards the eye
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.vstack([right, up, back]), 0.5 * (V.min(0) + V.max(0))


def render(stem: str, az: float, el: float, w: int, h: int, zmax: float = np.inf, fov_deg: float = 14.0,
           margin: float = 0.05) -> Image.Image:
    V, N, F, col, met, pid = load_scene(CAD_DIR / f"{stem}.glb")
    keep = V[F, 2].min(axis=1) <= zmax  # optional clip plane for close-ups (mm); pixels above it are not drawn
    R, c = _camera(V, az, el)
    cam = (V - c) @ R.T  # x right, y up, z back
    W, H = w * SS, h * SS
    # Perspective with a long lens: fit the projected extent into the frame
    span = cam[:, :2].max(0) - cam[:, :2].min(0)
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    dist = 0.5 * max(span) * f + cam[:, 2].max()
    z = dist - cam[:, 2]  # distance in front of the eye
    px, py = cam[:, 0] / z, -cam[:, 1] / z
    # Fit the projected (visible) extent exactly into the frame
    vis = np.zeros(len(V), dtype=bool)
    vis[F[keep].ravel()] = True
    vis &= V[:, 2] <= zmax + 0.02 * span.max()
    lo = np.array([px[vis].min(), py[vis].min()])
    hi = np.array([px[vis].max(), py[vis].max()])
    s = min(W * (1 - 2 * margin) / (hi[0] - lo[0]), H * (1 - 2 * margin) / (hi[1] - lo[1]))
    sx = W / 2 + s * (px - 0.5 * (lo[0] + hi[0]))
    sy = H / 2 + s * (py - 0.5 * (lo[1] + hi[1]))
    vn = N @ R.T
    depth = np.full((H, W), np.inf)
    nbuf = np.zeros((H, W, 3))
    fid = np.full((H, W), -1, dtype=np.int64)
    _raster(sx, sy, z, V[:, 2].copy(), float(zmax), F, vn, W, H, depth, nbuf, fid)

    hit = fid >= 0
    n = nbuf / np.maximum(np.linalg.norm(nbuf, axis=2, keepdims=True), 1e-9)
    n[n[..., 2] < 0] *= -1.0  # double-sided: face the camera
    base = np.where(hit[..., None], col[np.maximum(fid, 0)], 0.0)
    metal = np.where(hit, met[np.maximum(fid, 0)], 0.0)

    view = np.array([0.0, 0.0, 1.0])
    rgb = base * (0.30 + 0.12 * n[..., 1:2])  # hemispheric ambient (sky above, ground below)
    for L, I in ((np.array([-0.45, 0.65, 0.62]), 0.78), (np.array([0.7, 0.15, 0.45]), 0.32)):
        L = L / np.linalg.norm(L)
        ndl = np.clip(n @ L, 0, None)[..., None]
        hv = (L + view) / np.linalg.norm(L + view)
        spec = np.clip(n @ hv, 0, None)[..., None] ** (24 + 40 * metal[..., None])
        tint = metal[..., None] * base + (1 - metal[..., None]) * 1.0
        rgb = rgb + I * (ndl * base * (1 - 0.35 * metal[..., None]) + (0.18 + 0.45 * metal[..., None]) * spec * tint)
    rgb = np.clip(rgb, 0, 1)

    # Feature lines: depth jumps, part boundaries and creases
    d = np.where(hit, depth, depth[hit].max() * 1.5 if hit.any() else 0)
    pidb = np.where(hit, pid[np.maximum(fid, 0)], -1)
    line = np.zeros((H, W), dtype=bool)
    tol = 0.004 * (np.nanmax(d[hit]) - np.nanmin(d[hit]) + 1e-9) if hit.any() else 1.0
    for dy, dx in ((0, 1), (1, 0)):
        a_ = (slice(0, H - dy), slice(0, W - dx))
        b_ = (slice(dy, H), slice(dx, W))
        both = hit[a_] & hit[b_]
        jump = (np.abs(d[a_] - d[b_]) > tol) | (pidb[a_] != pidb[b_]) | (both & ((n[a_] * n[b_]).sum(-1) < 0.6))
        line[a_] |= jump
    rgb[line] *= 0.28

    # Background: soft vertical gradient
    gy = np.linspace(0, 1, H)[:, None, None]
    bg = (1 - gy) * np.array([0.965, 0.972, 0.982]) + gy * np.array([0.86, 0.88, 0.905])
    img = np.where(hit[..., None] | line[..., None], rgb, bg)
    img = img.reshape(h, SS, w, SS, 3).mean(axis=(1, 3))
    return Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Module 6: Rendering CAD stills ===")
    index = {}
    for name, (stem, az, el, w, h, zmax, cap) in VIEWS.items():
        img = render(stem, az, el, w, h, zmax)
        img.save(OUT_DIR / f"{name}.png", optimize=True)
        index[name] = {"file": f"{name}.png", "source": f"{stem}.glb", "caption": cap, "size_px": [w, h]}
        print(f"  {name:15s} <- {stem}.glb  ({w}x{h})")
    (OUT_DIR / "renders.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
