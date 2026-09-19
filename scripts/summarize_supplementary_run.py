"""Check and summarize committed CSV outputs from the 2026-09-19 run.

This reads archived results. It does not reproduce model inference or figures.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median


RUN = Path(__file__).resolve().parents[1] / "runs" / "20260919-200658"
TABLES = RUN / "tables"
EXPECTED_ROWS = {
    "checkpoint_inventory": 35,
    "detector_floor": 24,
    "efficiency": 7,
    "localization_accuracy": 7,
    "localization_report": 21,
    "paired_differences": 21,
    "panel_per_frame": 21,
    "robustness": 5040,
    "strain_reference_space": 27,
    "subpixel_precision": 3024,
    "training_curves_summary": 10,
}


def read_table(name: str) -> list[dict[str, str]]:
    with (TABLES / f"{name}.csv").open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    tables = {name: read_table(name) for name in EXPECTED_ROWS}
    for name, expected in EXPECTED_ROWS.items():
        actual = len(tables[name])
        if actual != expected:
            raise ValueError(f"{name}.csv: expected {expected} rows; found {actual}")
    print(f"Recorded CSV row counts verified for {len(tables)} tables.")

    loc = tables["localization_accuracy"]
    assert {row["fold"] for row in loc} == {"2"}
    assert {row["n_frames"] for row in loc} == {"40"}
    print("\nFold 2 localization (40 recorded frames per setting):")
    for row in loc:
        print(
            f"  {row['condition']:8s} {row['arch']:10s} "
            f"F1={float(row['f1']):.4f} "
            f"RMSE={float(row['rmse_px']):.3f} px"
        )

    efficiency = tables["efficiency"]
    assert {row["fold"] for row in efficiency} == {"2"}
    print("\nFold 2 device measurements (batch size 1):")
    for row in efficiency:
        print(
            f"  {row['condition']:8s} {row['arch']:10s} "
            f"median={float(row['ms_median_b1']):.2f} ms "
            f"VRAM={float(row['vram_mb_b1']):.1f} MB"
        )

    shift = tables["subpixel_precision"]
    print("\nDirect AtomSegNet controlled shift, median row RMSE:")
    for dose in (50, 200, 1000):
        for source in ("raw", "denoised"):
            values = [
                float(row["rmse"])
                for row in shift
                if row["condition"] == "no_n2v"
                and row["arch"] == "AtomSegNet"
                and row["source"] == source
                and int(row["dose"]) == dose
            ]
            if len(values) != 72:
                raise ValueError(f"Expected 72 shift rows for {dose}, {source}; found {len(values)}")
            print(f"  dose={dose:4d} source={source:8s} RMSE={median(values):.3f} px")

    paired = tables["paired_differences"]
    print("\nFive-fold paired IoU changes, N2V minus direct:")
    for arch in ("AtomSegNet", "UNetPP", "HRNet"):
        rows = [
            row for row in paired
            if row["metric"] == "iou"
            and row["a"] == f"{arch} with_n2v"
            and row["b"] == f"{arch} no_n2v"
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one paired IoU row for {arch}")
        print(f"  {arch:10s} delta={float(rows[0]['mean']):+.4f}")


if __name__ == "__main__":
    main()
