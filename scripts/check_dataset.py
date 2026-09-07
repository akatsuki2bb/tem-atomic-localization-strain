#!/usr/bin/env python3
"""Validate the expected TEM-ImageNet-v1.3 folder layout."""
from pathlib import Path
import os
import sys

cwd = Path.cwd().resolve()
project = Path(os.environ.get("TEM_PROJECT_ROOT", cwd))
data = Path(os.environ.get("TEM_DATA_ROOT", project / "data" / "TEM-ImageNet-v1.3"))
required = ["image", "noNoise", "circularMask", "gaussianMask", "position"]

print(f"Dataset root: {data}")
missing = []
for name in required:
    p = data / name
    ok = p.is_dir()
    count = sum(1 for x in p.iterdir() if x.is_file()) if ok else 0
    print(f"  {'OK' if ok else 'MISSING':7s} {name:14s} files={count}")
    if not ok:
        missing.append(name)

if missing:
    print("\nMissing required folders:", ", ".join(missing))
    sys.exit(1)
print("\nDataset layout looks valid.")
