"""Diff 02b verdicts: prior (HEAD) vs new (working tree)."""
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
p02b = prior[prior.feature_set_version == "v1_opening_drive_shock"].set_index(key)
n02b = new[new.feature_set_version == "v1_opening_drive_shock"].set_index(key)

print(f"prior 02b rows: {len(p02b)}")
print(f"new 02b rows:   {len(n02b)}")
print()

cols = [
    "n_train", "n_val", "n_test",
    "brier_test_baseline", "brier_test_candidate",
    "brier_improvement", "calibration_improvement",
    "passed_stability",
]

merged = p02b[cols].join(n02b[cols], lsuffix="_prior", rsuffix="_new", how="outer")

print("=" * 116)
print("Per-(feature, window) Brier comparison")
print("=" * 116)
header = (
    f"{'feature':<32}  {'window':<14} {'test':<5} "
    f"{'prior brier_imp':>16}  {'new brier_imp':>16}  "
    f"{'delta':>10}  {'prior P/F':>9}  {'new P/F':>9}"
)
print(header)
print("-" * len(header))

verdict_changes = []
for idx, row in merged.iterrows():
    feat, fsv, tw, ts = idx
    pri_imp = row["brier_improvement_prior"]
    new_imp = row["brier_improvement_new"]
    pri_pass = bool(row["passed_stability_prior"])
    new_pass = bool(row["passed_stability_new"])
    delta = (new_imp - pri_imp) if pd.notna(new_imp) and pd.notna(pri_imp) else float("nan")
    flag = " <-- VERDICT CHANGE" if pri_pass != new_pass else ""
    if pri_pass != new_pass and (feat, pri_pass, new_pass) not in [
        (vc[0], vc[3], vc[4]) for vc in verdict_changes
    ]:
        verdict_changes.append((feat, tw, ts, pri_pass, new_pass))
    print(
        f"{feat:<32}  {tw:<14} {ts!s:<5} "
        f"{pri_imp:>+16.6f}  {new_imp:>+16.6f}  "
        f"{delta:>+10.6f}  {('PASS' if pri_pass else 'FAIL'):>9}  "
        f"{('PASS' if new_pass else 'FAIL'):>9}{flag}"
    )

print()
print("=" * 116)
print("Per-feature stability summary")
print("=" * 116)
for feat in sorted(set(merged.index.get_level_values("feature"))):
    sub_p = p02b[p02b.index.get_level_values("feature") == feat]
    sub_n = n02b[n02b.index.get_level_values("feature") == feat]
    p_brier = (sub_p["brier_improvement"] > 0).sum()
    n_brier = (sub_n["brier_improvement"] > 0).sum()
    p_ece = (sub_p["calibration_improvement"] > 0).sum()
    n_ece = (sub_n["calibration_improvement"] > 0).sum()
    p_pass = bool(sub_p["passed_stability"].iloc[0]) if len(sub_p) else None
    n_pass = bool(sub_n["passed_stability"].iloc[0]) if len(sub_n) else None
    p_v = "PASS" if p_pass else "FAIL"
    n_v = "PASS" if n_pass else "FAIL"
    flag = " <-- VERDICT CHANGE" if p_pass != n_pass else ""
    print(
        f"  {feat:<32} prior {p_brier}/3 brier {p_ece}/3 ece ({p_v})  ->  "
        f"new {n_brier}/3 brier {n_ece}/3 ece ({n_v}){flag}"
    )

print()
print(f"verdict changes (unique features): "
      f"{len(set(vc[0] for vc in verdict_changes))}")

print()
print("=" * 116)
print("Test-set size deltas (n_test prior vs new)")
print("=" * 116)
for feat in sorted(set(n02b.index.get_level_values("feature"))):
    p_sub = p02b[p02b.index.get_level_values("feature") == feat]
    n_sub = n02b[n02b.index.get_level_values("feature") == feat]
    print(f"  {feat:<32}")
    for tw in ["2015-2020", "2015-2021", "2015-2022"]:
        try:
            p_n = int(p_sub.xs(tw, level="train_window")["n_test"].iloc[0])
            n_n = int(n_sub.xs(tw, level="train_window")["n_test"].iloc[0])
            d = n_n - p_n
            arrow = "  (no change)" if d == 0 else f"  (delta {d:+d})"
            print(f"    {tw}: prior n_test={p_n:,}  new n_test={n_n:,}{arrow}")
        except Exception as e:
            print(f"    {tw}: ERROR {e}")
