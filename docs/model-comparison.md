# Model comparison

## Current benchmark

This page summarizes the 14 September 2026 benchmark in [`../tables/benchmark_summary.csv`](../tables/benchmark_summary.csv). Each row contains five folds, and each fold was evaluated on 600 held-out frames.

| Condition | Architecture | Parameters (M) | IoU | Dice | PSNR (dB) | SSIM | ΔPSNR vs Gaussian (dB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Direct | AtomSegNet | 9.121 | **0.8763 ± 0.0026** | **0.9330** | 24.68 ± 0.73 | 0.706 | -1.93 |
| Direct | HRNet | 9.181 | 0.8629 ± 0.0019 | 0.9249 | 26.45 ± 0.34 | 0.801 | -0.16 |
| Direct | U-Net++ | 9.118 | 0.8601 ± 0.0071 | 0.9236 | 18.13 ± 8.14 | 0.615 | -8.48 |
| Direct | SwinUNet | 9.272 | 0.8547 ± 0.0041 | 0.9199 | **30.82 ± 0.35** | **0.917** | **+4.21** |
| N2V warm start | AtomSegNet | 9.121 | 0.8616 ± 0.0018 | 0.9244 | 20.98 ± 0.45 | 0.587 | -5.63 |
| N2V warm start | HRNet | 9.181 | 0.8248 ± 0.0064 | 0.9009 | 15.82 ± 0.78 | 0.445 | -10.79 |
| N2V warm start | U-Net++ | 9.118 | 0.8604 ± 0.0101 | 0.9236 | 15.17 ± 6.59 | 0.527 | -11.43 |

IoU uses a fixed threshold of 0.5. PSNR/SSIM use the normalized denoising output and `noNoise` target. The Gaussian reference is recomputed within each fold on the same inputs.

## What the comparison supports

### Segmentation

Direct AtomSegNet has the highest fixed-threshold IoU. Paired fold comparisons in the benchmark notebook give mean advantages of 0.0134 over direct HRNet, 0.0162 over direct U-Net++, and 0.0216 over direct SwinUNet. These comparisons are based on only five fold-level observations and should be treated as internal benchmark evidence, not a population-level claim.

### Denoising

Direct SwinUNet is the only configuration with a positive mean PSNR gain over the Gaussian σ = 1 reference. U-Net++ is particularly unstable across folds. Consequently, the CNN denoising heads should not be presented as successful denoisers on this snapshot even when their segmentation remains strong.

### N2V warm start

The paired fold effect on IoU is:

| Architecture | Mean ΔIoU, N2V − direct | N2V fold wins | Paired p-value |
|---|---:|---:|---:|
| AtomSegNet | -0.0147 | 0/5 | 4.92 × 10⁻⁶ |
| HRNet | -0.0382 | 0/5 | 6.02 × 10⁻⁵ |
| U-Net++ | +0.0003 | 3/5 | 0.945 |

The p-values are exploratory because there are only five folds. SwinUNet has no N2V run and is excluded from this contrast.

## Supporting analyses and their scope

These analyses demonstrate the measurement pipeline but are not used to establish the headline ranking.

| Artifact | Scope | Correct interpretation |
|---|---|---|
| [`subpixel_precision.csv`](../tables/subpixel_precision.csv) | 3 frames × 3 doses × 5 noise repeats × 8 shifts × 3 coordinate sources; one selected fold/configuration | Controlled shift sensitivity on selected examples |
| [`localization_report.csv`](../tables/localization_report.csv) | 3 demonstration frames per configuration | Spacing and centroid sanity check |
| [`centroid_metrics.csv`](../tables/centroid_metrics.csv) | First 40 validation candidates/fold; only 1–4 dense frames/fold pass the ≥20-column gate | Sparse supporting check, not a full localization benchmark |
| [`physics_all_frames.csv`](../tables/physics_all_frames.csv) | 60 dense frames per configuration from one selected fold | Lattice/strain pipeline operation |
| Experimental transfer files under [`../results/`](../results/) | Unlabeled images | Qualitative transfer and failure analysis, not accuracy |

Some committed tables contain angstrom or picometre columns calculated with a historical placeholder scale. Those columns are not physically validated and should not be quoted; use the corresponding pixel-unit columns.

## Bottom line

The strongest research-facing result is not “model X wins.” It is the demonstrated separation between image restoration and quantitative microscopy: direct SwinUNet gives the best intensity reconstruction, direct AtomSegNet gives the best segmentation, and the downstream coordinate tests require their own validation.
