"""The rebuilt outputs against the committed ones: the end of ./run_paper.sh, and of ./run_all.sh on the frozen texts.
    python checks/verify_outputs.py
Compares every committed file of general/outputs, general/plots and docs/paper/figures that differs from the last
commit, by what it holds rather than by its bytes:
  - PNG: the decoded pixels must be identical (the bytes of a PNG depend on the version of the image libraries);
  - CSV: the same rows, columns and text, and every number equal within 0.001 (absolute) or 0.1% (relative), the
    last digit of a rounded value that another platform or library version can move;
  - PDF: identical bytes except the creation date that matplotlib writes into it;
  - any other file: identical bytes.
Exits with 1, naming each file and the first difference, if anything else changed or a file is missing.
"""
import io
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["general/outputs", "general/plots", "docs/paper/figures"]
ATOL, RTOL = 1e-3, 1e-3


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def same_png(old, new):
    a, b = Image.open(io.BytesIO(old)), Image.open(io.BytesIO(new))
    if a.size != b.size:
        return f"size {a.size} -> {b.size}"
    pa, pb = np.asarray(a.convert("RGBA")), np.asarray(b.convert("RGBA"))
    n = int((pa != pb).any(axis=-1).sum())
    return f"{n:,} pixels differ" if n else None


def same_csv(old, new):
    a, b = (pd.read_csv(io.BytesIO(x), keep_default_na=False, dtype=str) for x in (old, new))
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return f"shape {a.shape} -> {b.shape} or other columns"
    for col in a.columns:
        x, y = pd.to_numeric(a[col], errors="coerce"), pd.to_numeric(b[col], errors="coerce")
        num = x.notna() & y.notna()
        text = ~num & (a[col] != b[col])
        if text.any():
            i = text.idxmax()
            return f"row {i}, {col!r}: {a.at[i, col]!r} -> {b.at[i, col]!r}"
        far = num & ~np.isclose(x.where(num, 0), y.where(num, 0), atol=ATOL, rtol=RTOL)
        if far.any():
            i = far.idxmax()
            return f"row {i}, {col!r}: {a.at[i, col]} -> {b.at[i, col]}"
    return None


def same_pdf(old, new):
    date = re.compile(rb"/CreationDate \(D:[^)]*\)")
    return None if date.sub(b"", old) == date.sub(b"", new) else "the bytes differ (not only the creation date)"


def main():
    changed = [p for p in git("diff", "--name-only", "HEAD", "--", *DIRS).decode().splitlines() if p]
    problems, close = [], []
    for p in changed:
        full = os.path.join(ROOT, p)
        if not os.path.exists(full):
            problems.append(f"{p}: missing")
            continue
        old, new = git("show", f"HEAD:{p}"), open(full, "rb").read()
        if p.endswith(".png"):
            why = same_png(old, new)
        elif p.endswith(".csv"):
            why = same_csv(old, new)
        elif p.endswith(".pdf"):
            why = same_pdf(old, new)
        else:
            why = "the bytes differ"
        (problems if why else close).append(f"{p}: {why}" if why else p)
    if close:
        print(f"{len(close)} files differ from the commit in their bytes only (same pixels, numbers within "
              f"{ATOL:g} or {RTOL:.1%}, or a PDF's creation date):")
        for p in close:
            print("  " + p)
    for p in problems:
        print("  CHANGED: " + p)
    print("outputs: same as committed" if not problems else f"outputs: {len(problems)} files CHANGED")
    return not problems


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
