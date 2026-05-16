"""Diff 02a verdicts: prior (HEAD) vs new (working tree)."""
import pandas as pd
from pathlib import Path
import subprocess

ROOT = Path(r"C:\Users\Alexander\Documents\CFB\CFBapp")

prior_path = ROOT / "research" / "results" / "_prior_feature_validation_HEAD.csv"
subprocess.run(
    ["git", "show", "HEAD:research/results/feature_validation.csv"],
    cwd=ROOT,
    stdout=open(prior_path, "wb"),
    check=True,
)

prior = pd.read_csv(prior_path)
new = pd.read_csv(ROOT / "research" / "results" / "feature_validation.csv")

key = ["feature", "feature_set_version", "train_window", "test_season"]

p02a = prior[prior.feature_set_version == "v1_baseline_efficiency_only"].set_index(key)
n02a = new[new.feature_set_version == "v1_baseline_efficiency_only"].set_index(key)

print(f"prior 02a rows: {len(p02a)}")
print(f"new 02a rows:   {len(n02a)}")
print()

cols = [
    "n_train", "n_val", "n_test",
    "brier_test_baseline", "brier_test_candidate",
    "brier_improvement", "calibration_improvement",
    "passed_stability",
]

merged = p02a[cols].join(n02a[cols], lsuffix="_prior", rsuffix="_new", how="outer")

print("=" * 110)
print("Per-(feature, window) Brier comparison")
print("=" * 110)

header = (
    f"{'feature':<26}  {'window':<14} {'test':<5} "
    f"{'prior brier_imp':>16}  {'new brier_imp':>16}  "
    f"{'delta':>10}  {'prior P/F':>9}  {'new P/F':>9}"
)
print(header)
print("-" * len(header))

verdict_change = []
for idx, row in merged.iterrows():
    feat, fsv, tw, ts = idx
    pri_imp = row["brier_improvement_prior"]
    new_imp = row["brier_improvement_new"]
    pri_pass = bool(row["passed_stability_prior"])
    new_pass = bool(row["passed_stability_new"])
    delta = (new_imp - pri_imp) if pd.notna(new_imp) and pd.notna(pri_imp) else float("nan")
    flag = " <-- VERDICT CHANGE" if pri_pass != new_pass else ""
    if pri_pass != new_pass:
        verdict_change.append((feat, tw, ts, pri_pass, new_pass))
    print(
        f"{feat:<26}  {tw:<14} {ts!s:<5} "
        f"{pri_imp:>+16.6f}  {new_imp:>+16.6f}  "
        f"{delta:>+10.6f}  {('PASS' if pri_pass else 'FAIL'):>9}  "
        f"{('PASS' if new_pass else 'FAIL'):>9}{flag}"
    )

print()
print("=" * 110)
print("Per-feature stability summary")
print("=" * 110)
for feat in sorted(set(merged.index.get_level_values("feature"))):
    sub_p = p02a[p02a.index.get_level_values("feature") == feat]
    sub_n = n02a[n02a.index.get_level_values("feature") == feat]
    p_brier_imp = (sub_p["brier_improvement"] > 0).sum()
    n_brier_imp = (sub_n["brier_improvement"] > 0).sum()
    p_pass = bool(sub_p["passed_stability"].iloc[0]) if len(sub_p) else None
    n_pass = bool(sub_n["passed_stability"].iloc[0]) if len(sub_n) else None
    p_verdict = "PASS" if p_pass else ("FAIL" if p_pass is False else "MISS")
    n_verdict = "PASS" if n_pass else ("FAIL" if n_pass is False else "MISS")
    flag = " <-- VERDICT CHANGE" if p_pass != n_pass else ""
    print(
        f"  {feat:<26} prior {p_brier_imp}/3 ({p_verdict})  ->  "
        f"new {n_brier_imp}/3 ({n_verdict}){flag}"
    )

print()
print(f"verdict changes: {len(verdict_change)}")

# Null counts (working-tree feature_matrix isn't on disk; report new only)
# But we can compute baseline brier (sanity) and approx null counts via n_test
print()
print("=" * 110)
print("Test-set size diffs (proxy for null count shifts)")
print("=" * 110)
for feat in sorted(set(n02a.index.get_level_values("feature"))):
    p_sub = p02a[p02a.index.get_level_values("feature") == feat]
    n_sub = n02a[n02a.index.get_level_values("feature") == feat]
    print(f"  {feat:<26}")
    for tw in ["2015-2020", "2015-2021", "2015-2022"]:
        try:
            p_n_test = int(p_sub.xs(tw, level="train_window")["n_test"].iloc[0])
            n_n_test = int(n_sub.xs(tw, level="train_window")["n_test"].iloc[0])
            d = n_n_test - p_n_test
            arrow = "  (no change)" if d == 0 else f"  (delta {d:+d})"
            print(f"    {tw}: prior n_test={p_n_test:,}  new n_test={n_n_test:,}{arrow}")
        except Exception as e:
            print(f"    {tw}: ERROR {e}")
