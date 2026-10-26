#!/usr/bin/env python3
"""Regenerate measured-data Oct 2026 TEM metrology SVG figures.

Run:
    python scripts/plot_oct2026_metrology.py

Uses only the committed CSV exports (not illustrative/synthetic curves).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "2026-10-quantitative-metrology"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_table(name: str):
    with (BASE / "tables" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


P = read_table("precision_all_models.csv")
S = read_table("lattice_strain_all_models.csv")
MODELS = [
    ("no_n2v", "AtomSegNet", "AtomSegNet direct", "#1c629c"),
    ("no_n2v", "HRNet", "HRNet direct", "#db8326"),
    ("no_n2v", "UNetPP", "U-Net++ direct", "#188a83"),
    ("no_n2v", "SwinUNet", "SwinUNet direct", "#7351b6"),
    ("with_n2v", "AtomSegNet", "AtomSegNet N2V", "#69a8dc"),
    ("with_n2v", "HRNet", "HRNet N2V", "#b76244"),
    ("with_n2v", "UNetPP", "U-Net++ N2V", "#d363ae"),
]


def get_strain(condition, model):
    return next(row for row in S if row["condition"] == condition and row["model"] == model)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def dose_jitter():
    fig, ax = plt.subplots(figsize=(12, 6))
    for condition, model, label, col in MODELS:
        rows = sorted(
            (r for r in P if r["condition"] == condition
             and r["model"] == model and r["source"] == "raw"),
            key=lambda r: float(r["D"]),
        )
        ax.plot([float(r["D"]) for r in rows],
                [float(r["jitter_pm"]) for r in rows],
                "o--" if condition == "with_n2v" else "o-",
                label=label, color=col, linewidth=2, markersize=5)
    ax.set(xscale="log", xlabel="Poisson count scale D (counts at unit intensity)",
           ylabel="RMS repeat jitter in x (pm)",
           title="Atomic-column localization repeatability: raw-weight coordinates")
    ax.set_xticks([5, 20, 100, 1000], labels=["5", "20", "100", "1000"])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "repeatability_vs_dose.svg")


def crlb_gap():
    fig, ax = plt.subplots(figsize=(11, 6))
    vals = [
        next(r for r in P if r["condition"] == condition
             and r["model"] == model and r["source"] == "raw"
             and int(r["D"]) == 1000)
        for condition, model, _, _ in MODELS
    ]
    x = np.arange(len(vals))
    y = np.array([float(r["jitter_to_crlb_ratio"]) for r in vals])
    lo = np.array([float(r["ratio_95ci_low"]) for r in vals])
    hi = np.array([float(r["ratio_95ci_high"]) for r in vals])
    ax.bar(x, y, color=[m[3] for m in MODELS], width=0.7)
    ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="none", ecolor="#333333", capsize=3)
    ax.axhline(1, linestyle="--", color="#111111", linewidth=1.2)
    ax.set(xticks=x, xticklabels=[m[2].replace(" ", "\\n") for m in MODELS],
           ylim=(0, 3), ylabel="Measured jitter / Poisson CRLB",
           title="High-count localization remains above the Poisson reference bound (D=1000)")
    ax.grid(axis="y", alpha=0.18)
    save(fig, "poisson_crlb_gap.svg")


def lattice_recovery():
    fig, ax = plt.subplots(figsize=(11, 6))
    vals = [float(get_strain(a, m)["lattice_found_pct"]) for a, m, _, _ in MODELS]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=[t[3] for t in MODELS], width=0.7)
    ax.set(xticks=x, xticklabels=[t[2].replace(" ", "\\n") for t in MODELS],
           ylim=(0, 103), ylabel="Lattice recovery (%)",
           title="Lattice recovered from detections: 8,947 eligible held-out frames")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.9, f"{v:.1f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    save(fig, "lattice_recovery.svg")


def strain_floor():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MODELS))
    rms = [float(get_strain(a, m)["strain_rms_exx_pct"]) for a, m, _, _ in MODELS]
    floor = [float(get_strain(a, m)["floor_exx_pct"]) for a, m, _, _ in MODELS]
    ax.bar(x - 0.2, rms, width=0.4, color=[t[3] for t in MODELS], label="Strain error RMS")
    ax.bar(x + 0.2, floor, width=0.4, facecolor="#c6d1dd", label="Localization-derived floor")
    ax.set(xticks=x, xticklabels=[t[2].replace(" ", "\\n") for t in MODELS],
           ylim=(0, 1.6), ylabel="Local normal strain xx error (%)",
           title="Strain field errors compared with the independent-coordinate-error floor")
    ax.grid(axis="y", alpha=0.18)
    ax.legend()
    save(fig, "strain_error_vs_floor.svg")



def calibrated_subpixel_diagnostics():
    """Bias and first-harmonic pixel locking on measured D=1000 raw-frame rows."""
    vals = [
        next(r for r in P if r["condition"] == cond and r["model"] == model
             and r["source"] == "raw" and int(r["D"]) == 1000)
        for cond, model, _, _ in MODELS
    ]
    x = np.arange(len(vals))
    labs = ["AS", "HR", "U++", "Swin", "AS+N2V", "HR+N2V", "U+++N2V"]
    fig, (ax, lockax) = plt.subplots(1, 2, figsize=(14, 6))
    ax.scatter(x - 0.11, [float(r["bias_x_px"]) for r in vals],
               s=48, color="#2e6cab", label="signed x bias", zorder=3)
    ax.scatter(x + 0.11, [float(r["bias_y_px"]) for r in vals],
               s=48, color="#e58b56", marker="s", label="signed y bias", zorder=3)
    ax.axhline(0, color="#555555", linestyle="--")
    ax.set(xticks=x, xticklabels=labs, ylim=(-0.04, 0.04),
           ylabel="Signed error against exact atoms (px)",
           title="Residual coordinate bias after calibration")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    lockax.bar(x, [float(r["pixel_locking_px"]) for r in vals],
               color=[m[3] for m in MODELS])
    lockax.set(xticks=x, xticklabels=labs, ylim=(0, 0.007),
               ylabel="First-harmonic pixel locking (px)",
               title="Pull toward pixel centres")
    lockax.grid(axis="y", alpha=0.2)
    fig.suptitle("Raw-frame metrology at D=1000: separate bias and pixel locking")
    save(fig, "residual_bias_and_pixel_locking_D1000.svg")


def raw_vs_reconstructed_jitter():
    """Distinguish N2V training warm start from the peak-fitting image source."""
    x = np.arange(len(MODELS))
    labs = [name for _, _, name, _ in MODELS]
    raw = [
        next(float(r["jitter_pm"]) for r in P
             if r["condition"] == cond and r["model"] == model
             and r["source"] == "raw" and int(r["D"]) == 1000)
        for cond, model, _, _ in MODELS
    ]
    reconstructed = [
        next(float(r["jitter_pm"]) for r in P
             if r["condition"] == cond and r["model"] == model
             and r["source"] == "denoised" and int(r["D"]) == 1000)
        for cond, model, _, _ in MODELS
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.vlines(x, raw, reconstructed, color="#9aabba", linewidth=2)
    ax.scatter(x - 0.08, raw, label="Raw frame", color="#2e6cab", s=70, zorder=3)
    ax.scatter(x + 0.08, reconstructed, label="Reconstructed frame",
               color="#e58b56", marker="s", s=70, zorder=3)
    ax.set(xticks=x, xticklabels=[q.replace(" ", "\\n") for q in labs],
           ylim=(0, 1.8), ylabel="RMS repeat jitter (pm)",
           title="Raw vs reconstructed image for localization (D=1000)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    save(fig, "raw_vs_reconstructed_jitter_D1000.svg")


if __name__ == "__main__":
    dose_jitter()
    crlb_gap()
    lattice_recovery()
    strain_floor()
    calibrated_subpixel_diagnostics()
    raw_vs_reconstructed_jitter()
    print(f"Saved six measured-data figures to {FIG}")
