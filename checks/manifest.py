"""Write data/<freeze>/manifest.json: for every file of the freeze, its SHA-256, size, number of records
and (for raw ADS pulls) the time it was saved. Run after a freeze is complete:
    python checks/manifest.py
A later run that finds a different checksum means the frozen data were changed."""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import FREEZE, DATA   # noqa: E402

files = {}
for path in sorted(glob.glob(os.path.join(DATA, "**", "*"), recursive=True)):
    if not os.path.isfile(path) or path.endswith("manifest.json"):
        continue
    rel = os.path.relpath(path, DATA)
    entry = {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "bytes": os.path.getsize(path)}
    if path.endswith(".json"):
        obj = json.load(open(path, encoding="utf-8"))
        if isinstance(obj, dict):
            entry["saved"] = obj.get("saved")
            entry["records"] = len(obj["docs"]) if "docs" in obj else len(obj.get("corpus_docs", obj))
    elif path.endswith(".csv"):
        entry["records"] = sum(1 for _ in open(path, encoding="utf-8")) - 1
    files[rel] = entry
json.dump({"freeze": FREEZE, "files": files}, open(os.path.join(DATA, "manifest.json"), "w"), indent=1)
print(f"manifest: {len(files)} files -> data/{FREEZE}/manifest.json")
