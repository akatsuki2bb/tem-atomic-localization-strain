#!/usr/bin/env python3
"""Validate the TEM-ImageNet-v1.3 subset used by the benchmark."""
from pathlib import Path
import hashlib
import os
import sys

cwd = Path.cwd().resolve()
project = Path(os.environ.get("TEM_PROJECT_ROOT", cwd))
data = Path(os.environ.get("TEM_DATA_ROOT", project / "data" / "TEM-ImageNet-v1.3"))
required = ["image", "noNoise", "circularMask", "gaussianMask"]
optional = ["position"]
expected_count = 14_364
expected_md5 = "34fc3dda6080f89bc90d49e36695cf24"

print(f"Dataset root: {data}")
missing = []
indices = {}
for name in required:
    p = data / name
    ok = p.is_dir()
    stems = {x.stem for x in p.iterdir() if x.is_file()} if ok else set()
    indices[name] = stems
    count = len(stems)
    print(f"  {'OK' if ok else 'MISSING':7s} {name:14s} files={count}")
    if not ok:
        missing.append(name)

for name in optional:
    p = data / name
    count = sum(1 for x in p.iterdir() if x.is_file()) if p.is_dir() else 0
    print(f"  {'OPTIONAL' if p.is_dir() else 'ABSENT':7s} {name:14s} files={count}")

if missing:
    print("\nMissing required folders:", ", ".join(missing))
    sys.exit(1)

paired = sorted(set.intersection(*(indices[name] for name in required)))
digest = hashlib.md5("|".join(paired).encode()).hexdigest()
print(f"\nPaired stems: {len(paired)}")
print(f"Ordered-stem MD5: {digest}")

if len(paired) != expected_count or digest != expected_md5:
    print(
        "WARNING: this dataset snapshot differs from the committed benchmark "
        f"({expected_count} pairs, MD5 {expected_md5})."
    )
else:
    print("Dataset snapshot matches the committed benchmark.")
