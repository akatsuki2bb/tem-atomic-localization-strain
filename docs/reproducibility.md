# Reproducibility

## Recommended environment

- Python 3.10–3.11
- PyTorch 2.1+
- CUDA-capable NVIDIA GPU for full training

Install with either `requirements.txt` or `environment.yml`.

## Environment variables

The notebook defaults to repository-relative paths. Optional overrides:

### Linux/macOS

```bash
export TEM_PROJECT_ROOT="$PWD"
export TEM_DATA_ROOT="$PWD/data/TEM-ImageNet-v1.3"
export TEM_RESULTS_ROOT="$PWD/results_local"
export TEM_REAL_DATA="/path/to/experimental_tifs"
```

### Windows PowerShell

```powershell
$env:TEM_PROJECT_ROOT = (Get-Location).Path
$env:TEM_DATA_ROOT = "$env:TEM_PROJECT_ROOT\data\TEM-ImageNet-v1.3"
$env:TEM_RESULTS_ROOT = "$env:TEM_PROJECT_ROOT\results_local"
$env:TEM_REAL_DATA = "D:\path\to\experimental_tifs"
```

## Reproducibility cautions

1. Keep fold membership image-level and deterministic.
2. Do not compare incomplete architecture-condition combinations as if they were full 5-fold results.
3. Real-image transfer is currently unlabeled. Consensus is not accuracy.
4. The simulated 0.5 segmentation threshold is not calibrated for the real-image coverage distribution.
5. Report sub-pixel **precision/repeatability** separately from absolute localization RMSE.
6. Large checkpoints and caches are intentionally ignored by Git.
