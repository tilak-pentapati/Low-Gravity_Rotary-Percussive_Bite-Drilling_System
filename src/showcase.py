"""
Shared numbers for the portfolio showcase (site landing page and the one-page PDF handout).

Everything is read from the pipeline exports (exports/data/*.json); nothing here is a result in its own right.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exports" / "data"

# Icebreaker drill field-test result, "1-1-100-100" (Zacny et al. 2013): 1 m, ~1 h, < 100 N WOB, ~100 W
ICEBREAKER = {"depth_m": 1.0, "time_h": 1.0, "wob_n": 100.0, "power_w": 100.0}


def load() -> Dict[str, Any]:
    j = lambda n: json.loads((DATA / f"{n}.json").read_text())  # noqa: E731
    des, abcd, sol, man = j("design_derivations"), j("abcd_summary"), j("solar_trade"), j("cad_manifest")
    led = des["ledger"]
    E = abcd["environments"]
    m = des["mast"]
    f = {
        "des": des, "abcd": abcd, "sol": sol, "man": man, "led": led, "E": E,
        "V": lambda k: led[k]["value"],
        "n_ledger": len(led),
        "lander_kg": abcd["lander_mass_kg"],
        "mast_band_margin_pct": 100 * m["band_margin"],
        "mast_f_lo": min(m["f_deployed_hz"]), "mast_f_hi": max(m["f_deployed_hz"]),
        "tilt_margin": m["lander"]["env"]["moon"]["margin"],
        "mast_candidates": (m["candidates_feasible"], m["candidates_total"]),
        "solar_sel": sol["labels"][sol["selected"]],
        "solar": {k: sol["configs"][k] for k in sol["configs"]},
        "solar_labels": sol["labels"],
        "n_parts": man["assemblies"]["full_drill_rig"]["parts"],
    }
    return f


def benchmark_rows(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows of (metric, Icebreaker, Moon, Mars, verdict) for the 1-1-100-100 comparison."""
    mo, ma = f["E"]["moon"], f["E"]["mars"]
    ib = ICEBREAKER

    def verdict(vals, limit, lower_is_better=True):
        ok = [(v <= limit) if lower_is_better else (v >= limit) for v in vals]
        if all(ok):
            return "meets"
        return "Moon meets, Mars above" if ok[0] else ("Mars meets, Moon above" if ok[1] else "above")

    return [
        {"metric": "Depth", "unit": "m", "ib": ib["depth_m"], "moon": mo["A_reached_depth_m"],
         "mars": ma["A_reached_depth_m"], "fmt": "{:.2f}",
         "verdict": verdict([mo["A_reached_depth_m"], ma["A_reached_depth_m"]], ib["depth_m"], False)},
        {"metric": "Mission time", "unit": "h", "ib": ib["time_h"],
         "moon": mo["B_total_operational_duration_hr"], "mars": ma["B_total_operational_duration_hr"], "fmt": "{:.2f}",
         "verdict": verdict([mo["B_total_operational_duration_hr"], ma["B_total_operational_duration_hr"]], ib["time_h"])},
        {"metric": "WOB set-point", "unit": "N", "ib": ib["wob_n"], "moon": mo["C_operational_wob_setpoint_n"],
         "mars": ma["C_operational_wob_setpoint_n"], "fmt": "{:.0f}",
         "verdict": verdict([mo["C_operational_wob_setpoint_n"], ma["C_operational_wob_setpoint_n"]], ib["wob_n"])},
        {"metric": "Drilling power", "unit": "W", "ib": ib["power_w"], "moon": mo["D_mean_drilling_power_w"],
         "mars": ma["D_mean_drilling_power_w"], "fmt": "{:.0f}",
         "verdict": verdict([mo["D_mean_drilling_power_w"], ma["D_mean_drilling_power_w"]], ib["power_w"])},
    ]
