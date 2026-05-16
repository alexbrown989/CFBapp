"""One-shot probe of the new CFBD key.

Issues a single cheap GET to /conferences, prints ALL response headers so
we can see what CFBD exposes for rate-limit / quota. Does NOT touch the
research cache (no new cache files written). Cost: 1 CFBD call.
"""
from __future__ import annotations

import os
import pathlib
import sys

import httpx
from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "backend" / ".env"
assert ENV_PATH.exists(), f"missing {ENV_PATH}"
load_dotenv(ENV_PATH)

key = os.environ.get("CFBD_API_KEY", "")
assert key, "CFBD_API_KEY not set after load_dotenv"
print(f"key length: {len(key)} chars; first 4 = {key[:4]!r}; last 4 = {key[-4:]!r}")

URL = "https://apinext.collegefootballdata.com/conferences"
print(f"\nGET {URL}")
r = httpx.get(URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}, timeout=60)
print(f"status: {r.status_code}")
print(f"elapsed: {r.elapsed.total_seconds()*1000:.0f}ms")
print(f"content-bytes: {len(r.content):,}")

print("\n--- response headers ---")
for k, v in r.headers.items():
    print(f"  {k}: {v}")

# Anything that looks rate-limit-y, surface explicitly.
print("\n--- rate-limit-looking headers ---")
kw = ("ratelimit", "rate-limit", "quota", "remaining", "limit", "x-rl", "retry-after")
hits = [(k, v) for k, v in r.headers.items() if any(s in k.lower() for s in kw)]
if hits:
    for k, v in hits:
        print(f"  {k}: {v}")
else:
    print("  (none — CFBD v2 may not expose quota headers; will need dashboard check)")

if r.status_code != 200:
    print(f"\n[FAIL] non-200 response: {r.text[:500]!r}")
    sys.exit(1)

n_conferences = len(r.json())
print(f"\n[ok] auth succeeded, body has {n_conferences} conferences")
