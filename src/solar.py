"""
Module 0b: Solar geometry, array-configuration trade study and array power model.

For each environment the sun's elevation/azimuth are computed over one full day
(lunar synodic day / Mars sol). Five array configurations of the same four
0.68 m^2 panels are ray-traced against each other, the deployed drill mast,
latch posts and lander deck (envelopes from src/design_mast.py) for self-shading, with cell temperature from a radiative (+ convective on Mars)
balance and a linear temperature coefficient:

  A  fixed horizontal          D  1-axis azimuth tracking, vertical wings
  B  fixed vertical, 2 wings   E  2-axis tracking (azimuth + elevation) wings
     facing +/-X
  C  fixed vertical, 4 faces (+/-X, +/-Y)

The configuration with the highest day-averaged electrical output (both bodies
weighted equally) is selected and used by the state machine and CAD.

Irradiance model:
  beam    G_b = S_eff exp(-tau m),  m = 1/sin(h) capped at 40 (Mars); tau = 0 on the Moon
  diffuse G_d = 0.5 S_eff sin(h) (1 - exp(-tau m)) x (1 + cos beta)/2   (isotropic sky, Mars only)
  S_eff  = S_1AU / r^2, r from the Mars orbit at the chosen season Ls
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.design import EXPORTS_DATA_DIR, SIGMA_SB, derive_all, load_system_params

MOON_SYNODIC_S = 29.530589 * 86400.0
MARS_SOL_S = 88775.244

CONFIG_LABELS = {
    "A_fixed_horizontal": "Fixed horizontal (cells up)",
    "B_fixed_vertical_2wings": "Fixed vertical, 2 wings facing +/-X",
    "C_fixed_vertical_4faces": "Fixed vertical, 4 faces (+/-X, +/-Y)",
    "D_track_azimuth_vertical": "1-axis azimuth tracking, vertical",
    "E_track_2axis": "2-axis tracking (azimuth + elevation)",
}


def mars_sun_distance_au(env: Dict[str, Any]) -> float:
    a, e = float(env["orbit_semi_major_au"]), float(env["orbit_eccentricity"])
    nu = math.radians(float(env["season_ls_deg"]) - float(env["perihelion_ls_deg"]))
    return a * (1 - e * e) / (1 + e * math.cos(nu))


def solar_constant_eff(env_id: str, env: Dict[str, Any]) -> float:
    s = float(env["solar_constant_w_m2"])
    return s / mars_sun_distance_au(env) ** 2 if env_id == "mars" else s


def declination_deg(env_id: str, env: Dict[str, Any]) -> float:
    if env_id == "mars":
        return math.degrees(math.asin(math.sin(math.radians(env["obliquity_deg"])) * math.sin(math.radians(env["season_ls_deg"]))))
    # South-pole summer: sun at -dec (southern hemisphere) -> positive elevation
    return -float(env["sun_declination_deg"]) * (1 if env["latitude_deg"] < 0 else -1)


def day_length_s(env_id: str) -> float:
    return MOON_SYNODIC_S if env_id == "moon" else MARS_SOL_S


def sun_vector(env_id: str, env: Dict[str, Any], t_s: float) -> Tuple[float, float, np.ndarray]:
    """Elevation (deg), azimuth (deg from +Y=north, toward +X=east) and ENU unit vector at time t."""
    lat = math.radians(float(env["latitude_deg"]))
    dec = math.radians(declination_deg(env_id, env))
    hour0 = float(env.get("local_solar_hour_start", 12.0))
    hang = math.radians((hour0 - 12.0) * 15.0) + 2 * math.pi * t_s / day_length_s(env_id)
    sin_h = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hang)
    h = math.asin(max(-1.0, min(1.0, sin_h)))
    # ENU components
    e = -math.cos(dec) * math.sin(hang)
    n = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(hang)
    u = sin_h
    v = np.array([e, n, u])
    v /= np.linalg.norm(v)
    az = math.degrees(math.atan2(e, n)) % 360.0
    return math.degrees(h), az, v


@dataclass
class Panel:
    center: np.ndarray
    normal: np.ndarray
    up: np.ndarray
    width: float
    height: float

    @property
    def right(self) -> np.ndarray:
        return np.cross(self.up, self.normal)

    def samples(self, n_w: int = 12, n_h: int = 24) -> np.ndarray:
        u = (np.arange(n_w) + 0.5) / n_w - 0.5
        v = (np.arange(n_h) + 0.5) / n_h - 0.5
        uu, vv = np.meshgrid(u * self.width, v * self.height)
        return self.center + uu.reshape(-1, 1) * self.right + vv.reshape(-1, 1) * self.up


def _ray_hits_panel(origins: np.ndarray, d: np.ndarray, pn: Panel) -> np.ndarray:
    denom = float(np.dot(d, pn.normal))
    if abs(denom) < 1e-9:
        return np.zeros(len(origins), dtype=bool)
    t = ((pn.center - origins) @ pn.normal) / denom
    hit = origins + t[:, None] * d
    rel = hit - pn.center
    return (t > 1e-6) & (np.abs(rel @ pn.right) <= pn.width / 2) & (np.abs(rel @ pn.up) <= pn.height / 2)


def _ray_hits_box(origins: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) > 1e-12, 1.0 / d, 1e12)
    t1 = (lo - origins) * inv
    t2 = (hi - origins) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    return (tmax >= np.maximum(tmin, 1e-6))


_GEOMETRY: Dict[int, Dict[str, Any]] = {}


def array_geometry(p: Dict[str, Any]) -> Dict[str, Any]:
    """Wing stations and shading envelopes derived with the mast (cached per parameter set)."""
    if id(p) not in _GEOMETRY:
        _GEOMETRY[id(p)] = derive_all(p)["mast"]["solar"]
    return _GEOMETRY[id(p)]


def build_panels(cfg: str, p: Dict[str, Any], sun: np.ndarray, elev_deg: float, geo: Optional[Dict[str, Any]] = None) -> List[Panel]:
    sa = p["power_system"]["solar_array"]
    geo = geo or array_geometry(p)
    w, h = float(sa["wing_width_m"]), float(sa["wing_height_m"])
    sep, yw = 2 * float(geo["wing_x_m"]), float(geo["wing_y_m"])
    z0 = float(geo["wing_z0_m"])
    zc = z0 + h / 2
    up_z = np.array([0.0, 0.0, 1.0])
    panels: List[Panel] = []
    half_panel_h = h / 2  # each wing = 2 panels stacked; treat a wing as one rectangle

    horiz = np.array([sun[0], sun[1], 0.0])
    horiz = horiz / (np.linalg.norm(horiz) + 1e-12)
    if cfg == "A_fixed_horizontal":
        for sx in (-1, 1):
            panels.append(Panel(np.array([sx * sep / 2, yw, z0]), up_z, np.array([0.0, 1.0, 0.0]), w, h))
    elif cfg == "B_fixed_vertical_2wings":
        for sx in (-1, 1):
            panels.append(Panel(np.array([sx * sep / 2, yw, zc]), np.array([sx, 0.0, 0.0]), up_z, w, h))
    elif cfg == "C_fixed_vertical_4faces":
        r = sep / 2
        for nrm in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = np.array([nrm[0], nrm[1], 0.0])
            panels.append(Panel(n * r + np.array([0, yw, z0 + half_panel_h / 2]), n, up_z, w, half_panel_h * 1.0))
    elif cfg == "D_track_azimuth_vertical":
        for sx in (-1, 1):
            panels.append(Panel(np.array([sx * sep / 2, yw, zc]), horiz, up_z, w, h))
    elif cfg == "E_track_2axis":
        for sx in (-1, 1):
            up = np.cross(sun, np.cross(up_z, sun))
            up = up / (np.linalg.norm(up) + 1e-12)
            panels.append(Panel(np.array([sx * sep / 2, yw, zc]), sun.copy(), up, w, h))
    return panels


def cell_temperature(g_abs: float, p: Dict[str, Any], env_id: str, env: Dict[str, Any], tilt_deg: float) -> float:
    sa = p["power_system"]["solar_array"]
    eps = float(sa["emissivity_front"]) + float(sa["emissivity_back"])
    vf_sky = (1 + math.cos(math.radians(tilt_deg))) / 2
    t_env4 = vf_sky * float(env["sky_temp_k"]) ** 4 + (1 - vf_sky) * float(env["initial_surface_temp_k"]) ** 4
    h_c = 2 * float(p["actuators"]["avionics_and_thermal"]["mars_convection_coeff_w_m2k"]) if env_id == "mars" else 0.0
    t_air = float(env["initial_surface_temp_k"])
    lo, hi = 20.0, 600.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        q = eps * SIGMA_SB * (t ** 4 - t_env4) + h_c * (t - t_air)
        if q > g_abs:
            hi = t
        else:
            lo = t
    return lo


def array_power(cfg: str, p: Dict[str, Any], env_id: str, env: Dict[str, Any], t_s: float,
                include_shading: bool = True, geo: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """geo: hardware geometry (design["mast"]["solar"]); defaults to the one derived from p."""
    geo = geo or array_geometry(p)
    sa = p["power_system"]["solar_array"]
    elev, az, sun = sun_vector(env_id, env, t_s)
    if elev <= 0.0:
        return {"power_w": 0.0, "elev_deg": elev, "az_deg": az, "shaded_frac": 0.0, "cell_temp_k": float(env["sky_temp_k"])}
    s_eff = solar_constant_eff(env_id, env)
    tau = float(env.get("optical_depth", 0.0))
    m = min(40.0, 1.0 / max(math.sin(math.radians(elev)), 1e-3))
    g_b = s_eff * math.exp(-tau * m)
    g_dh = 0.5 * s_eff * math.sin(math.radians(elev)) * (1 - math.exp(-tau * m))
    panels = build_panels(cfg, p, sun, elev, geo)
    eta28 = float(sa["cell_efficiency_28c"]) * float(sa["packing_factor"])
    beta_t = float(sa["temp_coeff_rel_per_k"])
    alpha = float(sa["absorptance"])
    total_area = float(sa["num_panels"]) * float(sa["panel_area_m2"])
    area_each = total_area / len(panels)
    boxes = [(np.array(lo), np.array(hi)) for lo, hi in geo["boxes_m"]]
    p_tot, shaded_acc = 0.0, 0.0
    for i, pn in enumerate(panels):
        cos_i = float(np.dot(pn.normal, sun))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, pn.normal[2]))))
        g_d = g_dh * (1 + math.cos(math.radians(tilt))) / 2
        frac_lit = 1.0
        if cos_i > 0 and include_shading:
            pts = pn.samples()
            blocked = np.zeros(len(pts), dtype=bool)
            for lo, hi in boxes:
                blocked |= _ray_hits_box(pts, sun, lo, hi)
            for j, other in enumerate(panels):
                if j != i:
                    blocked |= _ray_hits_panel(pts, sun, other)
            frac_lit = 1.0 - float(blocked.mean())
        g = max(0.0, cos_i) * g_b * frac_lit + g_d
        t_c = cell_temperature(alpha * g, p, env_id, env, tilt)
        eta = eta28 * (1 + beta_t * (t_c - 301.15))
        p_tot += area_each * eta * g
        shaded_acc += (1 - frac_lit) / len(panels)
    return {"power_w": p_tot, "elev_deg": elev, "az_deg": az, "shaded_frac": shaded_acc,
            "cell_temp_k": t_c}


def trade_study(p: Optional[Dict[str, Any]] = None, n_samples: int = 96) -> Dict[str, Any]:
    p = p or load_system_params()
    out: Dict[str, Any] = {"configs": {}, "labels": CONFIG_LABELS}
    for cfg in CONFIG_LABELS:
        row = {}
        for env_id, env in p["environments"].items():
            day = day_length_s(env_id)
            ts = np.linspace(0, day, n_samples, endpoint=False)
            pw = [array_power(cfg, p, env_id, env, float(t)) for t in ts]
            vals = np.array([x["power_w"] for x in pw])
            row[env_id] = {
                "mean_w": round(float(vals.mean()), 1),
                "min_w": round(float(vals.min()), 1),
                "max_w": round(float(vals.max()), 1),
                "mean_shaded_pct": round(100 * float(np.mean([x["shaded_frac"] for x in pw])), 1),
            }
        out["configs"][cfg] = row
    score = {c: v["moon"]["mean_w"] + v["mars"]["mean_w"] for c, v in out["configs"].items()}
    out["selected"] = max(score, key=score.get)
    out["sun"] = {}
    for env_id, env in p["environments"].items():
        out["sun"][env_id] = {
            "solar_constant_eff_w_m2": round(solar_constant_eff(env_id, env), 1),
            "declination_deg": round(declination_deg(env_id, env), 2),
            "day_length_h": round(day_length_s(env_id) / 3600.0, 2),
        }
    # First-order bound for comparison: P = A eta S (normal incidence, no packing, temperature, shading or dust)
    sa = p["power_system"]["solar_array"]
    area = float(sa["num_panels"]) * float(sa["panel_area_m2"])
    out["first_order_estimate_w"] = {e: round(area * float(sa["cell_efficiency_28c"]) * s["solar_constant_eff_w_m2"], 1)
                                     for e, s in out["sun"].items()}
    return out


def main() -> None:
    EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = trade_study()
    (EXPORTS_DATA_DIR / "solar_trade.json").write_text(json.dumps(res, indent=2))
    print("=== Module 0b: Solar array configuration trade (day-mean electrical W) ===")
    for cfg, v in res["configs"].items():
        print(f"  {CONFIG_LABELS[cfg]:42s} Moon {v['moon']['mean_w']:7.1f} W (shade {v['moon']['mean_shaded_pct']:4.1f}%) | "
              f"Mars {v['mars']['mean_w']:7.1f} W (shade {v['mars']['mean_shaded_pct']:4.1f}%)")
    print(f"  Selected: {CONFIG_LABELS[res['selected']]}")


if __name__ == "__main__":
    main()
