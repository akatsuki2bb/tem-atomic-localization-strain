# Reproducibility

This repository is reproducible at two different levels: the committed result snapshot can be checked from a clean clone, while full model evaluation requires external data and checkpoints.

## Recorded benchmark snapshot

| Field | Value |
|---|---|
| Source upload | commit `44c366d91f71446854bc3b2a7ecaae6f637dc344` |
| Evaluation date | 14 September 2026 |
| Paired frames | 14,364 at 256 × 256 px |
| Ordered-stem MD5 | `34fc3dda6080f89bc90d49e36695cf24` |
| Fold rule | `KFold(5, shuffle=True, random_state=42)` |
| Headline sample | 600 held-out images per fold |
| Mask threshold | 0.5 |
| Gaussian reference | σ = 1 |
| Checkpoints discovered | 35: 30 CNN and 5 direct-training SwinUNet |
| Recorded evaluation runtime | PyTorch 2.11.0+cu128, NVIDIA RTX 3070 Ti Laptop GPU |

The CNN checkpoints do not contain a validation-stem list or split hash. The evaluation notebook reconstructs the same split rule used in the training notebook, but cannot prove checkpoint/split identity from checkpoint metadata alone.

## Level 1: reproduce the portfolio summary

No dataset or GPU is required. The primary source is [`../tables/benchmark_summary.csv`](../tables/benchmark_summary.csv).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/plot_model_comparison.py
```

The script validates the seven expected architecture-condition rows and writes `assets/model_comparison.png`.

### Inspect the later supplementary archive

The supplied [19 September run](../runs/20260919-200658/) is preserved in full: `manifest.json`, `logs/events.jsonl`, 11 CSV tables, nine figures, and three image panels. To verify table row counts and recalculate selected CSV-based numbers without weights or a GPU:

```bash
python scripts/summarize_supplementary_run.py
```

The [run index](../runs/20260919-200658/README.md) lists the exact coverage of each table. The script reads archived CSVs; it does not redo inference, bootstrap intervals, training, or figure generation. The manifest records Windows 11, Python 3.14.2, PyTorch 2.11.0+cu128, NumPy 2.5.1 and an RTX 3070 Ti Laptop GPU for this follow-up. Its `updated` timestamp is stale relative to the event log, which records writes through 22:06:45. `strain_reference_space.csv` was overwritten and only its final 27-row version survives.

## Level 2: validate the dataset layout

Download TEM-ImageNet-v1.3 separately, then expose its root:

```bash
export TEM_PROJECT_ROOT="$PWD"
export TEM_DATA_ROOT="/path/to/TEM-ImageNet-v1.3"
python scripts/check_dataset.py
```

Required folders are `image/`, `noNoise/`, `circularMask/`, and `gaussianMask/`. The `position/` folder is optional for this benchmark. The data are not redistributed here.

## Level 3: rerun the benchmark

The evaluation notebook also needs the trained weights. The historical scan used this layout:

```text
<TEM_RESULTS_ROOT>/
├── without n2v/
│   ├── AtomSegNet/AtomSegNet_fold1.pt ... fold5.pt
│   ├── HRNet/HRNet_fold1.pt ... fold5.pt
│   └── UNetPP/UNetPP_fold1.pt ... fold5.pt
├── with n2v/
│   ├── AtomSegNet/AtomSegNet_fold1.pt ... fold5.pt
│   ├── HRNet/HRNet_fold1.pt ... fold5.pt
│   └── UNetPP/UNetPP_fold1.pt ... fold5.pt
└── SwinUNet/SwinUNet_fold1.pt ... fold5.pt
```

Set the paths and start the notebook from the repository root:

```bash
export TEM_PROJECT_ROOT="$PWD"
export TEM_DATA_ROOT="/path/to/TEM-ImageNet-v1.3"
export TEM_RESULTS_ROOT="/path/to/checkpoints-and-run-records"
export TEM_BENCHMARK_ROOT="$PWD/results_local/benchmark_v3"
export TEM_REAL_DATA="/path/to/experimental-tifs"   # optional

jupyter lab notebooks/03_benchmark_and_physics.ipynb
```

Equivalent PowerShell variables are `$env:TEM_PROJECT_ROOT`, `$env:TEM_DATA_ROOT`, `$env:TEM_RESULTS_ROOT`, `$env:TEM_BENCHMARK_ROOT`, and `$env:TEM_REAL_DATA`.

## Level 4: retrain

- [`../notebooks/01_cnn_training_pipeline.ipynb`](../notebooks/01_cnn_training_pipeline.ipynb) contains the three CNN experiments and N2V warm starts.
- [`../notebooks/02_swinunet_training.ipynb`](../notebooks/02_swinunet_training.ipynb) contains the direct-training SwinUNet experiment.
- Full five-fold training requires a CUDA-capable GPU and substantial runtime; it is not a CPU workflow.

The notebooks preserve the experimental record, but training is not yet packaged as a one-command CLI. For a publication archive, export immutable configuration files, checkpoint hashes, per-fold validation stems, and a fully pinned environment.

## Known reproducibility limits

1. The primary evaluation uses 600 images per fold, not every validation image.
2. The 600 images are a deterministic prefix of each reconstructed fold, not a separately seeded random sample.
3. Splitting is image-level; related simulation-family frames may cross folds.
4. N2V pretraining is transductive with respect to later validation images, although labels are not used.
5. CUDA benchmarking is enabled in the training notebooks, so reruns are not guaranteed to be bitwise identical.
6. Minimum package versions are provided, but no frozen lockfile or container image is committed.
7. Model weights and experimental images are not distributed in this repository.
8. The physical pixel calibration is unverified; use pixel-unit outputs unless an external calibration is supplied.
9. The 19 September archive contains no weight files, underlying simulation frames, raw per-epoch curves, or complete source code for all new supplementary procedures. The manifest points to local notebooks that are not included in the archive; the repository notebooks document related methods but cannot reproduce every new CSV and figure as supplied. Its fold-2 localization and efficiency results should not be generalized to five folds or other hardware.
