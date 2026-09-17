# Methods overview

## Research question

The project tests whether a model that improves TEM image appearance also improves atomic-column measurements. Denoising, segmentation, coordinate recovery, and strain diagnostics are therefore evaluated as separate tasks.

## Models and conditions

Four dual-head models produce a denoised image and atomic-column logits:

- AtomSegNet
- U-Net++
- HRNet
- SwinUNet

All models contain approximately 9.1 million parameters. AtomSegNet, U-Net++, and HRNet are evaluated both from direct supervised training and after a self-supervised Noise2Void warm start. The current SwinUNet experiment is direct-training only.

## Data and split

The development benchmark uses 14,364 paired 256 × 256 frames from TEM-ImageNet-v1.3:

- `image/`: noisy simulated input;
- `noNoise/`: denoising target;
- `circularMask/`: binary segmentation target;
- `gaussianMask/`: reference peaks used for coordinate analysis.

Five folds are regenerated with `KFold(n_splits=5, shuffle=True, random_state=42)` over the sorted paired basenames. The headline table evaluates a deterministic 600-image subset of each validation fold.

## Evaluation layers

### 1. Denoising

PSNR and SSIM compare the normalized denoising output with `noNoise`. A Gaussian filter with σ = 1 is evaluated on the same inputs as a simple reference. A neural denoiser is not treated as successful when it falls below that reference.

### 2. Segmentation

Pixel IoU, Dice, precision, and recall compare the probability map with `circularMask`. The headline IoU uses a fixed threshold of 0.5. A separate threshold sweep records the best pooled F1, but that optimized value is not substituted for the fixed-threshold headline metric.

### 3. Atomic-column localization

Local maxima are refined to sub-pixel centers and matched to peaks extracted from `gaussianMask`. Detection F1 and coordinate RMSE depend on the matching radius and on agreement between the two target encodings, so they are reported as supporting diagnostics rather than the main ranking.

### 4. Controlled shift recovery

Known fractional-pixel translations are applied before inference. After compensating for the imposed translation, residual bias, robust coordinate jitter, RMSE, and match fraction are measured. The current test uses three frames, three dose levels, five independent Poisson-noise realizations, eight shifts, and one selected checkpoint per configuration.

### 5. Lattice and strain diagnostics

Detected centers are fitted to a reference lattice. Residual displacement is propagated to local estimates of `exx`, `eyy`, `exy`, and rotation. The current batch contains 60 dense frames per configuration from one selected fold. Because the physical pixel scale is not verified and no imposed strain ground truth is used, this stage demonstrates pipeline operation rather than calibrated strain accuracy.

### 6. Experimental transfer

Experimental images have no atom labels. Transfer analysis therefore uses coverage, model disagreement, FFT/lattice preservation, and qualitative overlays. None of these is an experimental atom-detection accuracy.

## Interpretation rule

PSNR, segmentation overlap, coordinate repeatability, absolute coordinate error, and strain consistency answer different questions. They must not be collapsed into a single statement that one preprocessing method or architecture is “best.”
