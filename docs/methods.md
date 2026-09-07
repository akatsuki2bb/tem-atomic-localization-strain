# Methods overview

## Research question

The core comparison asks whether self-supervised Noise2Void preprocessing improves **quantitative atomic measurements**, not merely denoised-image appearance.

## Models

Three encoder-decoder backbones are compared under matched conditions:

- AtomSegNet
- UNet++
- HRNet

Each produces a denoised image head and atomic-mask logits. The benchmark compares direct training/inference with architecture-matched N2V warm starts / preprocessing.

## Simulated data

TEM-ImageNet-v1.3 pairs noisy frames with clean images, circular atom masks and Gaussian localization targets. Images are handled at 256 × 256 px. A uint8 memmap cache avoids repeated small-file reads during training.

## Validation layers

### 1. Segmentation

IoU, precision, recall and F1 are evaluated on held-out folds.

### 2. Absolute atomic localization

Probability maps are converted to atomic-column centers and matched to simulated localization ground truth. RMSE is reported in pixels.

### 3. Translation-consistency precision

Held-out frames are shifted by known offsets and re-inferred. After undoing the imposed shift, corresponding atom coordinates are compared. Pair RMSE is converted to an estimated single-shot precision using the independent-error approximation `single-shot ≈ pair RMSE / sqrt(2)`.

This is a **repeatability** measurement and should not be described as absolute sub-pixel accuracy.

### 4. Lattice and strain

Detected coordinates are robustly fitted to a reference lattice. Residual displacement is propagated to local fits of strain components `exx`, `eyy`, `exy`, rotation, dilatation and maximum shear. Quality diagnostics include valid-strain fraction, lattice-inlier fraction, lattice RMSE and local-fit residuals.

### 5. Experimental transfer

Because the real TEM set is unlabeled, no accuracy claim is made. Transfer is assessed using model consensus, disagreement, predicted coverage, bright/dark polarity, and preservation of lattice-frequency peaks in FFT space.

## Interpretation rule

Image fidelity metrics, segmentation overlap, absolute coordinate accuracy and coordinate repeatability answer different questions. They should not be collapsed into a single claim that denoising “helps” or “hurts.”
