# TEM denoising and atomic-column localization: run report

generated 2026-09-07 15:43:22 on NVIDIA GeForce RTX 3070 Ti Laptop GPU, torch 2.11.0+cu128 (CUDA 12.8), seed 42
dataset: 14364 paired samples at 256 px, 5-fold CV, conditions no_n2v, with_n2v, ~115.9 GPU-hours

## benchmark

| Condition | Arch | Folds | IoU | PSNR (dB) | SSIM | vs Gauss (dB) |
|---|---|---:|---|---|---|---|
| no_n2v | AtomSegNet | 5 | 0.824 ± 0.002 | 27.84 ± 0.21 | 0.869 | +3.48 |
| no_n2v | HRNet | 4 | 0.812 ± 0.002 | 28.10 ± 0.33 | 0.878 | +3.90 |
| no_n2v | UNetPP | 3 | 0.794 ± 0.014 | 22.07 ± 7.84 | 0.717 | -1.97 |
| with_n2v | UNetPP | 5 | 0.806 ± 0.015 | 16.02 ± 6.49 | 0.569 | -8.34 |
| with_n2v | AtomSegNet | 5 | 0.792 ± 0.001 | 25.89 ± 0.12 | 0.833 | +1.53 |
| with_n2v | HRNet | 1 | 0.773 | 18.82 | 0.699 | -5.87 |

## effect of N2V pretraining (paired by architecture and fold)

| Arch | Metric | Folds | without | with | delta | wins | p |
|---|---|---:|---|---|---|---|---|
| AtomSegNet | iou | 5 | 0.8236 | 0.7919 | -0.0317 ± 0.0012 | 0/5 | 0.000 |
| AtomSegNet | psnr | 5 | 27.8442 | 25.8903 | -1.9539 ± 0.2951 | 0/5 | 0.000 |
| AtomSegNet | ssim | 5 | 0.8691 | 0.8328 | -0.0363 ± 0.0027 | 0/5 | 0.000 |
| UNetPP | iou | 3 | 0.7943 | 0.8162 | +0.0220 ± 0.0153 | 3/3 | 0.131 |
| UNetPP | psnr | 3 | 22.0672 | 18.3530 | -3.7142 ± 8.7283 | 2/3 | 0.538 |
| UNetPP | ssim | 3 | 0.7169 | 0.6450 | -0.0719 ± 0.2322 | 2/3 | 0.645 |

A negative delta means the warm start made that metric worse on those folds.

## localization across folds

| Condition | Arch | Folds | F1 | RMSE (px) | Accepted fits |
|---|---|---:|---|---|---|
| no_n2v | AtomSegNet | 5 | 0.929 ± 0.010 | 1.13 ± 0.03 | 0.058 ± 0.011 |
| no_n2v | HRNet | 4 | 0.921 ± 0.018 | 1.12 ± 0.03 | 0.035 ± 0.010 |
| no_n2v | UNetPP | 3 | 0.918 ± 0.004 | 1.21 ± 0.01 | 0.067 ± 0.003 |
| with_n2v | AtomSegNet | 5 | 0.918 ± 0.010 | 1.24 ± 0.03 | 0.067 ± 0.021 |
| with_n2v | HRNet | 2 | 0.900 ± 0.013 | 1.25 ± 0.06 | 0.072 ± 0.032 |
| with_n2v | UNetPP | 5 | 0.921 ± 0.013 | 1.22 ± 0.06 | 0.056 ± 0.012 |

## sub-pixel precision (shift recovery on noisy input)

| Condition | Arch | Folds | Pair RMSE (px) | Single-shot (px) | Match rate |
|---|---|---:|---|---|---|
| no_n2v | AtomSegNet | 5 | 0.976 ± 0.163 | 0.690 ± 0.116 | 0.851 ± 0.102 |
| no_n2v | HRNet | 3 | 0.872 ± 0.141 | 0.617 ± 0.100 | 0.854 ± 0.087 |
| no_n2v | UNetPP | 3 | 0.721 ± 0.077 | 0.510 ± 0.055 | 0.884 ± 0.088 |
| with_n2v | AtomSegNet | 5 | 0.717 ± 0.060 | 0.507 ± 0.043 | 0.824 ± 0.087 |
| with_n2v | HRNet | 2 | 0.802 ± 0.120 | 0.567 ± 0.085 | 0.738 ± 0.234 |
| with_n2v | UNetPP | 5 | 0.776 ± 0.098 | 0.549 ± 0.070 | 0.872 ± 0.083 |

## caveats

1. Folds are seeded KFold over basenames; if filenames encode simulation series, sibling frames can occupy train and validation of the same fold.
2. cudnn.benchmark is enabled, so reruns are not bitwise reproducible.
3. All reported folds are held-out validation splits, not an independent test set.
4. No verified pixel calibration was supplied, so every spatial value is in pixels.
5. Both a from-scratch (no_n2v) and an N2V warm-started (with_n2v) arm were run; the paired contrast below isolates the warm start only on folds present in both arms.
6. N2V pretraining used all noisy images, including images that later served as validation in the CV folds (transductive; labels never seen).
7. Fold counts are unequal: {'no_n2v/HRNet': 4, 'no_n2v/UNetPP': 3, 'with_n2v/HRNet': 1} of 5. Cross-architecture and cross-condition means are provisional.
8. Metrics come from more than one evaluator (['backfill_eval', 'cv_log']); rows marked backfill_eval were measured after training on a subsample.
9. Mean PSNR is below a Gaussian sigma=1 filter for ['no_n2v/UNetPP', 'with_n2v/HRNet', 'with_n2v/UNetPP']; the denoising head did not converge on at least some folds and those means should not be quoted as denoising performance.
10. Precision values are pair RMSE from shifted-vs-unshifted comparisons; the single-shot estimate assumes independent errors at fractional pixel phase.

## artifacts

- [x] cv_results_all.csv
- [x] benchmark_comparison.csv
- [x] benchmark_summary.json
- [x] panel_samples.csv
- [x] precision_per_fold.csv
- [x] precision_across_folds.png
- [x] no_n2v/benchmark_bars.png
- [x] no_n2v/AtomSegNet/panels
- [x] no_n2v/UNetPP/panels
- [x] no_n2v/HRNet/panels
- [x] with_n2v/benchmark_bars.png
- [x] with_n2v/AtomSegNet/panels
- [x] with_n2v/UNetPP/panels
- [x] with_n2v/HRNet/panels
