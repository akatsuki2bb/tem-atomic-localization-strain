#!/usr/bin/env python3
"""Plot the *reported* 12,766-frame Atomap comparison from the committed summary.

The summary is a transcription of the later experiment's reported aggregate
metrics, not a replacement for per-frame detections or full benchmark reruns.

Run:
    python scripts/plot_atomap_benchmark_2026.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "2026-10-atomap-full-benchmark"
FIG = DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)

with (DIR / "tables" / "summary_reported.csv").open(
    newline="", encoding="utf-8"
) as fh:
    ROWS = list(csv.DictReader(fh))

R = {row["method"]: row for row in ROWS}
COLOUR = {"Atomap noisy": "#a1afbd", "AtomSegNet direct": "#2e75b5"}


def color_for(name: str) -> str:
    return COLOUR.get(name, "#81abc8")


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def f1_comparison() -> None:
    methods = [
        "Atomap noisy", "UNet++ direct", "SwinUNet direct",
        "UNet++ N2V", "HRNet direct", "AtomSegNet direct",
    ]
    y = [float(R[m]["reported_f1_detection_0p1_spacing"]) for m in methods]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(methods)), y,
                  color=[color_for(m) for m in methods])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(
        ylim=(0, 1.0), xticks=np.arange(len(methods)),
        xticklabels=[m.replace(" ", chr(10)) for m in methods],
        ylabel="Detection F1",
        title="Atomic-column detection: 12,766 held-out simulated frames",
    )
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "atomap_f1_full_benchmark.svg")


def rmse_comparison() -> None:
    methods = [
        "Atomap noisy", "UNet++ direct", "SwinUNet direct",
        "HRNet direct", "AtomSegNet direct",
    ]
    vals = [float(R[m]["reported_rmse_raw_px"]) for m in methods]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(np.arange(len(methods)), vals,
                   color=[color_for(m) for m in methods])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(
        xlim=(0, 2.0), yticks=np.arange(len(methods)),
        yticklabels=methods, xlabel="Matched coordinate RMSE (px)",
        title="Raw noisy-input localization (lower error is better)",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "atomap_rmse_full_benchmark.svg")


def four_metric_comparison() -> None:
    metrics = [
        ("Detection F1", "reported_f1_detection_0p1_spacing", "higher"),
        ("Raw coordinate RMSE (px)", "reported_rmse_raw_px", "lower"),
        ("Reference-aligned RMSE (px)",
         "reported_rmse_reference_aligned_px", "lower"),
        ("Coordinate scatter (px)", "reported_scatter_px", "lower"),
    ]
    a, b = R["Atomap noisy"], R["AtomSegNet direct"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, key, direction) in zip(axes.flat, metrics):
        values = [float(a[key]), float(b[key])]
        bars = ax.barh(["Atomap noisy", "AtomSegNet direct"], values,
                       color=["#a1afbd", "#2e75b5"])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.30)
        ax.set_title(f"{title} ({direction} is better)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Large-population Atomap / AtomSegNet localization benchmark")
    finish(fig, "atomap_vs_atomsegnet_four_metrics.svg")


if __name__ == "__main__":
    f1_comparison()
    rmse_comparison()
    four_metric_comparison()
    print(f"Wrote three figures to {FIG}")
