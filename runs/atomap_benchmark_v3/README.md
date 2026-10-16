# Atomap v3: atomic-column detection and localization benchmark

**Status:** Curated, traceable excerpt from the 16 October 2026 `atomap_benchmark_v3` output archive. The archive is a result export, **not** source code or a standalone rerunnable pipeline.

## Purpose and protocol

- Common evaluation population: 50 simulated TEM-image stems, fold 2; an additional 50 stems were allocated for calibration.
- Reference in the main table: Gaussian-mask local maxima, not independently validated physical atomic coordinates.
- Comparison: Atomap fitted peaks on noisy input and two image-only oracle controls vs direct or N2V-initialized AtomSegNet, HRNet and U-Net++.
- Raw coordinate state; detection matching within 0.10 of estimated neighbour spacing and localization matching within 0.25; fixed nominal bootstrap count 1,000.
- Report F1 (which includes false/missed detections) separately from RMSE (conditional on accepted matches). The units remain pixels.

## Main results: raw coordinates, Gaussian reference

| Method | Precision | Recall | F1 | Matched RMSE (px) | Localization coverage |
|---|---:|---:|---:|---:|---:|
| Atomap / noisy input | 0.185 | 0.409 | 0.255 | 2.155 | 55.6% |
| Atomap / noise-free control | 0.921 | 0.557 | 0.695 | 0.978 | 56.8% |
| Atomap / noise- and background-free control | 0.947 | 0.501 | 0.656 | 0.581 | 50.7% |
| AtomSegNet / direct | 0.801 | 0.900 | 0.847 | 1.187 | 99.6% |
| HRNet / direct | 0.803 | 0.900 | 0.849 | 1.237 | 99.2% |
| U-Net++ / direct | 0.801 | 0.896 | 0.846 | 1.247 | 99.2% |
| AtomSegNet / N2V | 0.762 | 0.850 | 0.804 | 1.339 | 98.3% |
| HRNet / N2V | 0.755 | 0.844 | 0.797 | 1.375 | 97.7% |
| U-Net++ / N2V | 0.806 | 0.904 | 0.852 | 1.239 | 99.1% |

Machine-readable exact-precision summary for these nine rows: [`main_results_gaussian_raw.csv`](main_results_gaussian_raw.csv). All rows use 50 evaluation frames and Gaussian reference. The CSV also includes per-frame bootstrap confidence intervals as reported by the original run.

## Interpretation

The network results outperform Atomap's **noisy-input configuration** in F1 and RMSE here. Among direct networks, HRNet has the highest F1 (0.8486) and AtomSegNet has the lowest matched RMSE (1.1867 px). U-Net++ after N2V has the highest F1 among the network rows (0.8520), but these small differences are not proof of statistically significant superiority. Atomap's no-background/no-noise control has a smaller conditional RMSE (0.5806 px) while detecting a much smaller fraction of reference columns (50.7% localization coverage).

## Limitations and audit status

1. **No independent split re-audit.** The source manifest states that evaluation membership is inherited from the earlier localization pipeline.
2. **Coordinate convention not fully audited.** Legacy `store['peaks']` were used as raw network coordinates; the manifest warns that a previous stage may already have calibrated them. Continuous `position` reference labels are deliberately disabled until axis order, origin and scale are verified.
3. **Calibration is asymmetric.** Translation calibration could be fitted for Atomap controls but not for any CNN configuration; a calibrated-vs-raw comparison would be misleading.
4. **Not an independent physical validation.** Images and references are simulated. The manifest gives no verified picometre-per-pixel scale.
5. **Uncertainty is conditional.** Confidence bounds resample frames, not necessarily independent physical structures or acquisition sessions, and do not propagate calibration-estimation uncertainty.
6. **Population consistency.** SwinUNet fold 4 is excluded from this particular comparison because it uses a different fold/evaluation population.
7. **No rerun code in this archive.** Checkpoint retrieval, inference and Atomap benchmark code must be documented separately before claiming one-command reproducibility.

## Outputs in the original upload

The full supplied archive comprises a run manifest, 36-row summary (three reference types and raw/calibrated states), per-frame measurements, matched error rows, position-convention audit, gate sweep, calibration table, Atomap diagnostic logs, figure PNGs/SVGs and an intermediate position cache. **This GitHub folder contains only the main Gaussian/raw comparison and interpretive documentation.** The original archive is required for the remaining underlying rows and figures.

## References

- [Atomap project and API documentation](https://atomap.org/api_documentation.html)
- [Relevant article supplied in the run manifest](https://doi.org/10.1038/s41598-021-84499-w)

See the [main repository README](../../README.md) for training and model provenance.
