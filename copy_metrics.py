#!/usr/bin/env python3
"""
Copies output/results -> output/result_metrics, excluding predictions.npz files.
"""

import shutil
from pathlib import Path

src = Path("output/results")
dst = Path("output/result_metrics")

if not src.exists():
    raise FileNotFoundError(f"Source directory not found: {src}")

copied = 0
skipped = 0

for src_file in src.rglob("*"):
    if not src_file.is_file():
        continue

    if src_file.name == "predictions.npz":
        skipped += 1
        continue

    rel = src_file.relative_to(src)
    dst_file = dst / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    copied += 1

print(f"Done. Copied {copied} files, skipped {skipped} predictions.npz files.")
print(f"Output: {dst.resolve()}")
