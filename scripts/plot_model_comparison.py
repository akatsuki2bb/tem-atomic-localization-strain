#!/usr/bin/env python3
"""Regenerate the research-facing benchmark figure from the committed CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED = [
    ("no_n2v", "AtomSegNet"),
    ("no_n2v", "HRNet"),
    ("no_n2v", "UNetPP"),
    ("no_n2v", "SwinUNet"),
    ("with_n2v", "AtomSegNet"),
    ("with_n2v", "HRNet"),
    ("with_n2v", "UNetPP"),
]

COLORS = {
    "AtomSegNet": "#0072B2",
    "HRNet": "#009E73",
    "UNetPP": "#D55E00",
    "SwinUNet": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "tables" / "benchmark_summary.csv",
        help="Current benchmark summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "assets" / "model_comparison.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "condition",
        "arch",
        "folds",
        "iou",
        "iou_sd",
        "psnr_gain_vs_gauss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    keyed = frame.set_index(["condition", "arch"], drop=False)
    found = set(keyed.index)
    expected = set(EXPECTED)
    if found != expected:
        raise ValueError(
            "Unexpected benchmark grid. "
            f"Missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    if not (frame["folds"] == 5).all():
        raise ValueError("Every headline configuration must contain five folds.")
    return keyed.loc[EXPECTED].reset_index(drop=True)


def display_label(condition: str, architecture: str) -> str:
    model = "U-Net++" if architecture == "UNetPP" else architecture
    suffix = "direct" if condition == "no_n2v" else "N2V warm start"
    return f"{model} · {suffix}"


def render(frame: pd.DataFrame, output: Path) -> None:
    labels = [display_label(c, a) for c, a in zip(frame.condition, frame.arch)]
    colors = [COLORS[a] for a in frame.arch]
    alpha = [0.95 if c == "no_n2v" else 0.55 for c in frame.condition]
    y = list(range(len(frame)))

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig, (ax_iou, ax_psnr) = plt.subplots(
            1, 2, figsize=(13.2, 5.7), sharey=True, gridspec_kw={"wspace": 0.08}
        )

        for yi, row, color, opacity in zip(y, frame.itertuples(), colors, alpha):
            ax_iou.errorbar(
                row.iou,
                yi,
                xerr=row.iou_sd,
                fmt="o",
                color=color,
                alpha=opacity,
                markersize=7,
                capsize=3,
                linewidth=1.6,
            )
            ax_iou.text(row.iou + 0.0017, yi, f"{row.iou:.3f}", va="center", fontsize=8.5)

        bars = ax_psnr.barh(y, frame.psnr_gain_vs_gauss, color=colors, height=0.62)
        for bar, opacity, value in zip(bars, alpha, frame.psnr_gain_vs_gauss):
            bar.set_alpha(opacity)
            offset = 0.3 if value >= 0 else -0.3
            align = "left" if value >= 0 else "right"
            ax_psnr.text(value + offset, bar.get_y() + bar.get_height() / 2,
                         f"{value:+.2f}", va="center", ha=align, fontsize=8.5)

        ax_iou.set_yticks(y, labels)
        ax_iou.invert_yaxis()
        ax_iou.set_xlim(0.812, 0.884)
        ax_iou.set_xlabel("Validation IoU at threshold 0.5 (expanded scale)")
        ax_iou.set_title("Atomic-column segmentation")
        ax_iou.grid(axis="x", color="#d9d9d9", linewidth=0.7)

        ax_psnr.axvline(0, color="#333333", linewidth=1)
        ax_psnr.set_xlim(-12.7, 5.7)
        ax_psnr.set_xlabel("PSNR gain over Gaussian σ = 1 (dB)")
        ax_psnr.set_title("Image denoising")
        ax_psnr.grid(axis="x", color="#d9d9d9", linewidth=0.7)

        for axis in (ax_iou, ax_psnr):
            axis.axhline(3.5, color="#bdbdbd", linewidth=0.8)
            axis.set_axisbelow(True)

        fig.suptitle(
            "Model selection depends on the measurement target",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            "Mean across five folds; 600 held-out images per fold. "
            "Error bars show fold SD for IoU. Faded marks denote N2V warm starts.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        fig.subplots_adjust(left=0.25, right=0.98, top=0.87, bottom=0.15)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=190, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input)
    render(summary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
