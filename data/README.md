# Data

No dataset is bundled in this repository.

Download TEM-ImageNet-v1.3 from its [source repository](https://github.com/xinhuolin/TEM-ImageNet-v1.3), subject to the dataset's own terms.

Place TEM-ImageNet-v1.3 here by default:

```text
data/TEM-ImageNet-v1.3/
├── image/
├── noNoise/
├── circularMask/
└── gaussianMask/
```

Or set `TEM_DATA_ROOT` to an external location.

`position/` is optional for the current benchmark. Atomic-center references are extracted from `gaussianMask/`; the `position/` files, when present, encode lattice information rather than a simple per-atom XY table.

Experimental TIFFs can be stored outside the repository and pointed to with `TEM_REAL_DATA`.
