"""One-shot: parse the PyPI scikit-learn JSON dump and report 1.7.x stable releases per R21."""
from __future__ import annotations
import json, re, sys, urllib.request

URL = "https://pypi.org/pypi/scikit-learn/json"
req = urllib.request.Request(URL, headers={"User-Agent": "R21-PyPI-check/1.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    raw = resp.read().decode("utf-8")
print(f"=== fetched {len(raw):,} bytes from {URL} ===")
data = json.loads(raw)

print("=== info.version (latest overall as advertised by PyPI) ===")
print(data["info"]["version"])
print()

print("=== all 1.7.x and 1.8.x release keys (raw, includes any pre-releases) ===")
keys = sorted(v for v in data["releases"].keys() if v.startswith(("1.7", "1.8")))
for v in keys:
    files = data["releases"][v]
    if not files:
        print(f"  {v:<25} (no files uploaded; empty release placeholder)")
        continue
    first_upload = min(f["upload_time_iso_8601"] for f in files)
    is_yanked = any(f.get("yanked") for f in files)
    yank_reason = next((f.get("yanked_reason") for f in files if f.get("yanked")), None)
    print(f"  {v:<25} first_upload={first_upload}  yanked={is_yanked}  yank_reason={yank_reason}")
print()

def is_stable(v: str) -> bool:
    return re.fullmatch(r"\d+\.\d+\.\d+", v) is not None

stable_17 = sorted(
    (v for v in data["releases"].keys() if v.startswith("1.7.") and is_stable(v)),
    key=lambda s: list(map(int, s.split("."))),
)

print("=== stable 1.7.x only (no rc/a/b/dev/post in version) ===")
for v in stable_17:
    files = data["releases"][v]
    if not files:
        print(f"  {v:<10} (no files; skip)")
        continue
    first_upload = min(f["upload_time_iso_8601"] for f in files)
    is_yanked = any(f.get("yanked") for f in files)
    print(f"  {v:<10} first_upload={first_upload}  yanked={is_yanked}")
print()

print("=== latest stable 1.7.x (non-yanked) ===")
candidates = [v for v in stable_17 if not any(f.get("yanked") for f in data["releases"][v])]
if not candidates:
    print("  (none)")
    sys.exit(0)
latest = candidates[-1]
files = data["releases"][latest]
upload = min(f["upload_time_iso_8601"] for f in files)
print(f"  version: {latest}")
print(f"  first-file upload (release date): {upload}")
print(f"  yanked: {any(f.get('yanked') for f in files)}")
