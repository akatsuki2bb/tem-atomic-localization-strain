# Results snapshot

These CSV files reproduce the current research snapshot used in the top-level README.

**Important:** the architecture-condition matrix is incomplete: 24/30 models are present. Full 5-fold claims should only be made for combinations with `n_folds = 5`.

- `current_segmentation_summary.csv`: held-out detection and absolute localization
- `atomsegnet_shift_precision.csv`: completed five-fold AtomSegNet translation-consistency test
- `real_transfer_summary.csv`: unlabeled experimental transfer diagnostics
- `strain_preliminary.csv`: preliminary strain quality metrics from four frames in fold 1

The real-image metrics are diagnostics, not accuracy measurements, because the experimental set is unlabeled.
