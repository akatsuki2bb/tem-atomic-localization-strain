# Deep Learning for Quantitative TEM

**Atomic-column localization, lattice metrology, strain mapping, and simulation-to-experiment transfer.**

This project asks a microscopy question rather than a leaderboard question:

> **Can denoising and deep-learning segmentation improve quantitative atomic measurements in noisy TEM, or do they only make the images look cleaner?**

The workflow compares **AtomSegNet, UNet++ and HRNet**, trained with and without a self-supervised **Noise2Void (N2V)** warm start. The network output is not treated as the final result: predicted atomic-column probability maps are converted into **atomic coordinates, lattice parameters, displacement fields, strain tensors, rotation, dilatation and shear**.

```text
Noisy TEM image
      ↓
optional N2V warm start / denoising
      ↓
atomic-column probability map
      ↓
atomic coordinates
      ↓
lattice fit
      ↓
displacement field
      ↓
εxx, εyy, εxy, rotation, dilatation, shear
```

![Example segmentation output](assets/segmentation_example.png)

## Research design

- **14,364 paired simulated TEM frames**, cropped to 256×256 px
- **5-fold cross-validation**
- **3 architectures:** AtomSegNet, UNet++, HRNet
- **2 conditions:** direct training/inference and N2V warm start
- controlled image-shift experiments for **localization repeatability**
- downstream **lattice and strain reconstruction**
- transfer diagnostics on **28 experimental TEM images** using an ensemble of the currently available models

The current run represents about **115.9 GPU-hours** on an NVIDIA GeForce RTX 3070 Ti Laptop GPU (seed 42).

## Main findings

### 1. AtomSegNet is the strongest complete segmentation benchmark so far

Across all five completed no-N2V folds, AtomSegNet achieved:

| Metric | Result |
|---|---:|
| IoU | **0.824 ± 0.002** |
| Atomic-site F1 | **0.929 ± 0.010** |
| Precision | 0.897 |
| Recall | **0.963** |
| Absolute localization RMSE | **1.13 ± 0.03 px** |

The very small IoU spread across folds is important: the result is stable rather than being driven by one favorable split.

### 2. Better denoising does not automatically mean better TEM metrology

For AtomSegNet, N2V reduced conventional segmentation and absolute-localization performance:

| Metric | No N2V | With N2V |
|---|---:|---:|
| IoU | **0.8236** | 0.7919 |
| PSNR | **27.84 dB** | 25.89 dB |
| SSIM | **0.869** | 0.833 |
| Absolute localization RMSE | **1.13 px** | 1.24 px |

The paired AtomSegNet IoU change was **−0.0317 ± 0.0012 across 5/5 folds** (`p < 0.001`).

However, N2V improved **translation-consistency precision**:

| AtomSegNet precision test | No N2V | With N2V |
|---|---:|---:|
| Pair RMSE | 0.976 ± 0.163 px | **0.717 ± 0.060 px** |
| Estimated single-shot precision | 0.690 ± 0.116 px | **0.507 ± 0.043 px** |
| Match rate | **0.851** | 0.824 |

This is the central result of the project: **N2V can make predicted atomic coordinates more repeatable under controlled translations while simultaneously making the absolute segmentation/localization less accurate.**

The single-shot value is a **precision/repeatability estimate**, not an absolute-accuracy claim; it assumes independent errors between shifted image pairs.

![Shift-recovery precision across folds](results/localization/precision_across_folds.png)

### 3. Atomic coordinates can be propagated into physically useful TEM measurements

Predicted coordinates are fitted to a reference lattice and converted into local displacement and strain fields. Preliminary no-N2V strain diagnostics currently cover four representative frames from fold 1.

AtomSegNet gave the best overall downstream measurement reliability in this preliminary set:

- median **valid-strain fraction: 0.874**
- median **lattice inlier fraction: 0.910**
- median **displacement RMS: 0.671 px**
- median local-fit residual: **0.555 px**

The strain analysis is still preliminary and is not presented as a final architecture ranking until more folds and frames are processed.

![Preliminary strain diagnostics](results/strain/strain_across_folds.png)

### 4. Simulation-to-experiment transfer works structurally, but not yet as calibrated atom segmentation

The current transfer experiment applies **24 trained model checkpoints to 28 unlabeled experimental TEM images**.

The models produce spatially structured probability maps and retain the expected bright-feature polarity in essentially all model-image pairs. Most importantly, an identifiable **lattice-frequency FFT peak is preserved in 85% of model-image pairs**. Here, 85% means that the structural periodicity visible in the experimental image remains detectable in frequency space after the model/denoising pipeline; it is **not** an 85% segmentation accuracy.

At the same time, the predicted foreground coverage spans **0.102–0.984**, compared with about **0.137 in the simulated training domain**. The simulated `0.5` probability threshold therefore does **not** transfer directly to the experimental images.

The real-image outputs should currently be interpreted as **candidate atomic-feature probability maps with model uncertainty**, not validated experimental atom masks.

![Experimental transfer diagnostics](results/transfer/transfer_probability.png)

## Why this matters for TEM

The project separates quantities that are often collapsed into a single image-quality score:

1. **Segmentation quality:** does the predicted atomic-column mask overlap the simulated target?
2. **Absolute positional accuracy:** how close is the inferred atomic coordinate to ground truth?
3. **Localization precision:** does the same atom return to the same physical coordinate after a known image translation?
4. **Downstream metrology:** are the coordinates stable enough for lattice fitting, displacement analysis and strain reconstruction?
5. **Experimental transfer:** does the learned representation preserve real structural information outside the simulated training domain?

That distinction matters because a denoiser can produce a visually smoother TEM image while shifting the quantities that are actually used for quantitative microscopy.

## Current benchmark status

The full design contains **30 architecture-condition-fold combinations**. **24 are currently available; 6 remain.** Complete five-fold claims are therefore restricted to combinations for which all five folds exist.

| Condition | Architecture | Available folds |
|---|---|---:|
| no N2V | AtomSegNet | **5/5** |
| no N2V | HRNet | 4/5 |
| no N2V | UNet++ | 3/5 |
| with N2V | AtomSegNet | **5/5** |
| with N2V | HRNet | 2/5 localization checkpoints; fewer benchmark records currently summarized |
| with N2V | UNet++ | **5/5** |

See [`results/reports/final_report.md`](results/reports/final_report.md) for the machine-generated run report and caveats.

## Repository structure

```text
.
├── notebooks/
│   └── TEM_quantitative_pipeline.ipynb
├── scripts/
│   └── check_dataset.py
├── data/
│   └── README.md
├── results/
│   ├── reports/
│   ├── benchmark/
│   ├── localization/
│   ├── strain/
│   └── transfer/
├── docs/
│   ├── methods.md
│   └── reproducibility.md
├── assets/
├── requirements.txt
├── environment.yml
├── .env.example
└── .gitignore
```

Large raw arrays, checkpoints, datasets, caches and multi-megabyte per-atom/per-pair tables are intentionally not required for browsing the repository. The curated `results/` directory contains the principal summaries, figures and machine-generated report.

## Dataset

The simulated benchmark uses **TEM-ImageNet-v1.3**:

https://github.com/xinhuolin/TEM-ImageNet-v1.3

Expected local layout:

```text
data/TEM-ImageNet-v1.3/
├── image/
├── noNoise/
├── circularMask/
├── gaussianMask/
└── position/
```

`position/` contains lattice vectors rather than per-atom XY coordinates, so localization ground truth is extracted from `gaussianMask/` in the current evaluation pipeline.

The experimental TEM images are **not distributed in this repository**.

## Quick start

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/check_dataset.py
jupyter lab notebooks/TEM_quantitative_pipeline.ipynb
```

The full training benchmark is intended for a CUDA-capable NVIDIA GPU. CPU execution is suitable for code inspection and lightweight diagnostics, not for practical retraining of the entire benchmark.

## Reproducibility and limitations

The main limitations of the current research snapshot are deliberately kept visible:

- reported folds are held-out CV validation splits, not a separate independent test set;
- filename-level simulation series may place related sibling frames into different CV folds;
- N2V pretraining used all noisy images without labels, making that arm **transductive** with respect to the later validation images;
- several architecture-condition combinations still have incomplete fold counts;
- some UNet++/HRNet denoising heads are unstable and should not be quoted as general denoising benchmarks;
- no verified physical pixel calibration was supplied, so spatial results are reported in **pixels**;
- experimental images are unlabeled, so transfer metrics are **diagnostics rather than accuracy measurements**;
- CUDA benchmarking is enabled, so reruns are not guaranteed to be bitwise identical.

See [`docs/reproducibility.md`](docs/reproducibility.md) and the generated [`final_report.md`](results/reports/final_report.md) for details.

## Status

- [x] simulated TEM data pipeline
- [x] AtomSegNet / UNet++ / HRNet benchmarking
- [x] direct vs N2V-warm-start comparison
- [x] atomic-site detection and absolute localization
- [x] controlled shift-recovery precision analysis
- [x] lattice fitting and 2D strain pipeline
- [x] 24-model consensus analysis on 28 experimental images
- [ ] complete the remaining 6 model-fold runs
- [ ] expand strain validation across folds and frames
- [ ] calibrate/annotate an experimental subset before claiming real-image atom-detection accuracy

**Manuscript in preparation.**
