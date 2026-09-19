# Follow-up measurements · 19 September 2026

The [five-fold benchmark](model-comparison.md) remains the primary comparison. A later [archived run](../runs/20260919-200658/) adds selected-fold localization and efficiency checks, a controlled shift experiment, a synthetic perturbation grid, comparison panels, and strain diagnostics. All original tables, plots and panels from the supplied run are available in its directory. These observations do not constitute a second independent dataset or a complete rerun of the headline benchmark.

## Selected-fold localization and compute

These are **fold 2 only**. The localization CSV records 40 frames and a 3 px matching radius per configuration. Timings are batch-size-1 measurements on one RTX 3070 Ti Laptop GPU; the training-time field is empty.

| Direct model | Detection F1 | Matched-coordinate RMSE (px) | Median inference (ms/image) |
|---|---:|---:|---:|
| AtomSegNet | 0.9476 | 1.322 | 12.23 |
| U-Net++ | 0.9308 | 1.392 | 32.16 |
| HRNet | 0.9067 | 1.405 | 49.26 |
| SwinUNet | 0.9174 | 1.377 | 20.37 |

Source: [`localization_accuracy.csv`](../runs/20260919-200658/tables/localization_accuracy.csv) and [`efficiency.csv`](../runs/20260919-200658/tables/efficiency.csv). Both also contain the three N2V CNN settings. AtomSegNet's selected-fold detection F1 is highest in this table, but the 40-frame result cannot establish an all-fold coordinate ranking. The separation between the best five-fold PSNR (SwinUNet) and best five-fold segmentation IoU (AtomSegNet) remains the well-supported main finding.

## Controlled shifts and stress tests

The new [`subpixel_precision.csv`](../runs/20260919-200658/tables/subpixel_precision.csv) tests three selected frames, three simulated dose settings, three noise repeats, eight fractional shifts, all seven settings and two image sources (`raw` and `denoised`). For **direct AtomSegNet at dose 1000**, the median of 72 recorded row-level RMSE values is **0.273 px on raw images** and **0.410 px on denoised images**. These rows share frames and shift settings; they are a descriptive selected-example contrast, not 72 independent tests. At other doses and for other models the relationship varies; see the [full table](../runs/20260919-200658/tables/subpixel_precision.csv) and [figure](../runs/20260919-200658/figures/subpixel_precision.png). The much smaller [`detector_floor.csv`](../runs/20260919-200658/tables/detector_floor.csv) is a 24-row diagnostic and should not be confused with model localization accuracy.

The [`robustness.csv`](../runs/20260919-200658/tables/robustness.csv) grid has 20 stems × four dose values × three blur values × three tilt values × seven configurations (5,040 rows). Its [plot](../runs/20260919-200658/figures/robustness.png) shows internal perturbation response. Neither the physical meaning of each tilt value nor experimental-image performance can be established from the supplied outputs alone.

The three [side-by-side panels](../runs/20260919-200658/panels/) illustrate both cleaner-looking and visibly distorted reconstructions; [frame 00257](../runs/20260919-200658/panels/panel_00257.png) is a convenient example. They are selected examples, with per-frame metrics in [`panel_per_frame.csv`](../runs/20260919-200658/tables/panel_per_frame.csv), not a random sample that replaces the benchmark.

## Strain remains exploratory

The final [`strain_reference_space.csv`](../runs/20260919-200658/tables/strain_reference_space.csv) contains 27 local estimates and was overwritten during the run, according to the [event log](../runs/20260919-200658/logs/events.jsonl). The [strain map](../runs/20260919-200658/figures/strain_map.png) and [robustness diagnostic](../runs/20260919-200658/figures/strain_robust.png) illustrate the pipeline, but the archive provides no independently imposed deformation truth for validating strain accuracy. The recorded pixel calibration is unverified. Do not quote these outputs as physically calibrated strain measurements.

The [run index](../runs/20260919-200658/README.md) lists every original output and its scope. The [summary script](../scripts/summarize_supplementary_run.py) checks CSV row counts and recalculates the selected numbers above from the archived tables; full figure/inference regeneration requires external materials described in [reproducibility](reproducibility.md).
