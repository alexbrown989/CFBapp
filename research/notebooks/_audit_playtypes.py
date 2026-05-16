"""One-shot audit: enumerate distinct playType values across a sample
of cached /plays files to make sure the explosive-play classifier covers
the actual variants."""
import json
from collections import Counter
from pathlib import Path

cache = Path("research/data/cache")
counter: Counter = Counter()
sample_files = sorted(cache.glob("cfbd__plays__*.json"))[:30]
print(f"sampling {len(sample_files)} plays files")
for p in sample_files:
    data = json.loads(p.read_text())
    for play in data:
        pt = play.get("playType")
        if pt is not None:
            counter[pt] += 1

print(f"\ntotal distinct playType values: {len(counter)}")
print(f"top 30 by frequency:")
for pt, n in counter.most_common(30):
    print(f"  {pt!r:<40}  {n:>8,}")

print(f"\nall types containing 'pass' or 'rush' (case-insensitive):")
for pt in sorted(counter):
    if "pass" in pt.lower() or "rush" in pt.lower():
        print(f"  {pt!r:<40}  {counter[pt]:>8,}")
