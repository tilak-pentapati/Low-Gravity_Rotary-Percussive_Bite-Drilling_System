"""
Module 8: One-page design summary (A4 PDF portfolio handout), built from the same exports as the site.

Output: exports/handout/design_summary.pdf (+ a PNG preview for the site).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.showcase import ROOT, benchmark_rows, load  # noqa: E402

OUT_DIR = ROOT / "exports" / "handout"
RENDERS = ROOT / "exports" / "renders"
A4 = (210.0, 297.0)
INK, MUTED, ACCENT, MOON, MARS = "#0f172a", "#475569", "#0369a1", "#0284c7", "#ea580c"
SITE_NOTE = "Interactive digital twin, derivation ledger, CAD (STEP/GLB) and drawings on the project site."
AUTHOR = "Jay Tilak P."


def _img(ax, name: str) -> None:
    ax.imshow(Image.open(RENDERS / f"{name}.png"))
    ax.axis("off")


def _para(fig, x, y, text, width, size=7.6, color=INK, weight="normal", lh=1.38) -> float:
    """Wrapped text block at (x, y) mm from the top-left; returns the y below it."""
    lines = []
    for p in text.split("\n"):
        lines += textwrap.wrap(p, width) or [""]
    for i, ln in enumerate(lines):
        fig.text(x / A4[0], 1 - (y + i * size * lh * 0.3528) / A4[1], ln, fontsize=size, color=color, weight=weight,
                 va="top", ha="left")
    return y + len(lines) * size * lh * 0.3528


def main() -> None:
    f = load()
    V, E = f["V"], f["E"]
    mo, ma = E["moon"], E["mars"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))

    # Header band
    fig.patches.append(plt.Rectangle((0, 1 - 30 / A4[1]), 1, 30 / A4[1], transform=fig.transFigure, color=INK))
    fig.text(12 / A4[0], 1 - 10 / A4[1], "Autonomous Rotary-Percussive Bite Drill for the Moon and Mars",
             fontsize=14, color="white", weight="bold", va="center")
    fig.text(12 / A4[0], 1 - 17.5 / A4[1],
             "1 m ice-bearing samples from a 45 kg lander, with no drilling fluid: physics-based design, mission simulation and CAD",
             fontsize=8, color="#cbd5e1", va="center")
    fig.text(12 / A4[0], 1 - 24 / A4[1], f"{AUTHOR}  ·  Model-based systems engineering in Python (build123d, numpy)",
             fontsize=7.5, color="#94a3b8", va="center")

    # Hero render + two details
    _img(fig.add_axes([0.03, 1 - 118 / A4[1], 0.60, 84 / A4[1]]), "rig_deployed")
    _img(fig.add_axes([0.63, 1 - 76 / A4[1], 0.17, 42 / A4[1]]), "head_section")
    _img(fig.add_axes([0.81, 1 - 76 / A4[1], 0.17, 42 / A4[1]]), "bit_closeup")
    fig.text(0.715, 1 - 78 / A4[1], "Drill head section", fontsize=6.5, color=MUTED, ha="center", va="top")
    fig.text(0.895, 1 - 78 / A4[1], "WC bit crown", fontsize=6.5, color=MUTED, ha="center", va="top")
    _img(fig.add_axes([0.63, 1 - 118 / A4[1], 0.35, 36 / A4[1]]), "rig_stowed")
    fig.text(0.805, 1 - 119 / A4[1], "Stowed for launch", fontsize=6.5, color=MUTED, ha="center", va="top")

    # Benchmark table
    y = 126
    fig.text(12 / A4[0], 1 - y / A4[1], "Results vs the Icebreaker drill field test (\"1-1-100-100\")", fontsize=9.5,
             weight="bold", color=ACCENT, va="top")
    y += 6
    cols = [12, 92, 122, 146, 170]
    for x, h in zip(cols, ["Metric", "Icebreaker", "Moon", "Mars", "vs benchmark"]):
        fig.text(x / A4[0], 1 - y / A4[1], h, fontsize=7.2, weight="bold", color=MUTED, va="top")
    for r in benchmark_rows(f):
        y += 4.6
        vals = [f"{r['metric']} ({r['unit']})", r["fmt"].format(r["ib"]), r["fmt"].format(r["moon"]),
                r["fmt"].format(r["mars"]), r["verdict"]]
        for x, v, c in zip(cols, vals, [INK, INK, MOON, MARS, INK]):
            fig.text(x / A4[0], 1 - y / A4[1], v, fontsize=7.4, color=c, va="top")
    y += 6
    y = _para(fig, 12, y, f"Same order of magnitude as the closest field-tested prototype (Icebreaker) with no coefficient fitted to the benchmark. "
              f"Mars WOB ({ma['C_operational_wob_setpoint_n']:.0f} N) is above Icebreaker's 100 N because the set-point is "
              f"lander weight / 1.4 and Mars gravity is 2.3x the Moon's.", 125, 6.9, MUTED)

    # Design decisions (left column) and limitations (right column)
    y0 = y + 4
    fig.text(12 / A4[0], 1 - y0 / A4[1], "Key design decisions and the physics behind them", fontsize=9.5,
             weight="bold", color=ACCENT, va="top")
    items = [
        ("Percussion, not more push.", f"The lander can only press {V('wob_lift_off_moon_n'):.0f} N on the Moon before lifting off; "
         f"rotary cutting of ice-cemented regolith needs more. A cam-and-spring hammer gives {V('impact_energy_j'):.2f} J at "
         f"{V('blow_frequency_hz'):.0f} Hz ({V('impact_peak_force_n') / 1e3:.1f} kN peak, stress-wave model)."),
        ("Bite drilling, not continuous augering.", f"Without fluid, flights cannot lift cuttings at a safe speed "
         f"(lift ratio {V('conveying_lift_ratio_moon'):.2f} < 1 on the Moon). Flutes hold one 10 cm bite "
         f"(choking at {V('choking_onset_bite_m') * 100:.1f} cm); a passive brush strips them into that bite's own sealed cup on a carousel."),
        ("Carbide chisel bit, not PDC.", f"Two WC-10Co inserts, -10° rake, R{V('bit_edge_radius_mm'):.1f} hone sized by Hertz "
         f"contact on a basalt clast (FOS {V('bit_edge_contact_fos'):.1f}); diamond would chip under impact and overheat dry."),
        ("Ice-preserving thermal control.", f"Bit temperature limit {V('critical_bit_temp_moon_k'):.0f} K (Moon, sublimation) / "
         f"{V('critical_bit_temp_mars_k'):.0f} K (Mars, melt); peak simulated {mo['max_bit_temp_k']:.0f} / {ma['max_bit_temp_k']:.0f} K."),
        ("Deployable side-mounted mast.", f"{V('mast_section')} spine chosen from "
         f"{f['mast_candidates'][1]} candidates; stowed {V('mast_stowed_f1_hz'):.0f} Hz (≥ 50 Hz launch), SMA hold-downs."),
        ("Tracking solar wings.", f"Sun ~1.5° above the lunar horizon: {f['solar_sel'].lower()} wings give "
         f"{f['solar']['E_track_2axis']['moon']['mean_w']:.0f} W (Moon) / {f['solar']['E_track_2axis']['mars']['mean_w']:.0f} W (Mars)."),
    ]
    y = y0 + 6
    for head, body in items:
        y = _para(fig, 12, y, head, 70, 7.4, INK, "bold", 1.25)
        y = _para(fig, 12, y, body, 70, 7.0, MUTED, "normal", 1.3) + 1.6

    fig.text(116 / A4[0], 1 - y0 / A4[1], "Honest limitations / next steps", fontsize=9.5, weight="bold",
             color=ACCENT, va="top")
    lims = [
        f"Deployed mast mode {f['mast_f_lo']:.0f}-{f['mast_f_hi']:.0f} Hz sits only {f['mast_band_margin_pct']:.0f} % from the "
        f"blow-rate band edge: a stiffer spine or tuned blow rate is the first change to test.",
        f"Side mount: lander tips before it lifts; tilt margin {f['tilt_margin']:.2f} vs 1.4 for a centred drill.",
        f"Rotating mast mass {V('mast_rotating_mass_kg'):.0f} kg drives a {V('deploy_actuator_torque_nm'):.0f} N m deploy actuator.",
        f"Brushing takes {V('transfer_duration_s'):.0f} s per bite ({mo['B_time_breakdown_min']['TRANSFER']:.0f} min per mission), "
        f"the largest single time item after warm-up.",
        "Rock-cutting and heat-partition coefficients come from literature ranges; sensitivity sweeps are shown, "
        "but cryo-vacuum drilling tests are needed to calibrate them.",
    ]
    y = y0 + 6
    for t in lims:
        y = _para(fig, 116, y, "•  " + t, 54, 7.0, MUTED, "normal", 1.3) + 1.4

    y = max(y, 0) + 3
    fig.text(116 / A4[0], 1 - y / A4[1], "How it was built", fontsize=9.5, weight="bold", color=ACCENT, va="top")
    _para(fig, 116, y + 6, f"One YAML parameter set (every input tagged as environment, design choice, coefficient or criterion, "
          f"with sources) feeds a derivation ledger of {f['n_ledger']} checked quantities, a 1 Hz mission simulation, "
          f"and {f['n_parts']}-part parametric CAD with STEP export. Change an input and every number, model and drawing "
          f"on the site regenerates; automated tests guard the physics.", 56, 7.0, MUTED, "normal", 1.3)

    fig.text(12 / A4[0], 8 / A4[1], SITE_NOTE, fontsize=6.8, color=MUTED, va="center")
    fig.savefig(OUT_DIR / "design_summary.pdf")
    fig.savefig(OUT_DIR / "design_summary.png", dpi=110)
    plt.close(fig)
    print("=== Module 8: one-page design summary -> design_summary.pdf ===")


if __name__ == "__main__":
    main()
