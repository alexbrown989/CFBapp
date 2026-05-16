"""Compare Brier improvement by test-season fold across PASS features.

Reads research/results/feature_validation.csv (committed working tree).

For each feature with passed_stability==True after the canonical walk-forward:
  pivot test_season (2022, 2023, 2024) -> brier_improvement.

Highlights whether 2024 folds are systematically weaker than 2022/2023 across
the 24 PASS features (21 pre-02e + 3 red-zone Failure v1_red_zone_failure).

Diagnostic-only / untracked per user protocol.
"""
from __future__ import annotations

import pathlib

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FV = REPO_ROOT / "research" / "results" / "feature_validation.csv"

S_THRESH = 0.005


def main() -> None:
    fv = pd.read_csv(FV, keep_default_na=False)
    fv["brier_improvement"] = fv["brier_improvement"].astype(float)
    fv["passed_stability"] = fv["passed_stability"].map(
        lambda x: str(x).strip().lower() in ("true", "1", "yes")
    )
    pass_feats = fv[fv["passed_stability"]][["feature", "feature_set_version"]].drop_duplicates()
    n_feat = len(pass_feats)
    print(f"PASS features (unique feature name): {n_feat}")

    rz_mask = fv["passed_stability"] & (fv["feature_set_version"] == "v1_red_zone_failure")
    pre_mask = fv["passed_stability"] & (fv["feature_set_version"] != "v1_red_zone_failure")
    rz = sorted(fv[rz_mask]["feature"].unique().tolist())
    pre = sorted(fv[pre_mask]["feature"].unique().tolist())
    print(f"  pre-02e (by name): {len(pre)}")
    print(f"  02e (v1_red_zone_failure): {len(rz)}")

    wide_rows: list[dict] = []
    for feat in sorted(pass_feats["feature"].unique()):
        sub = fv[(fv["feature"] == feat) & fv["passed_stability"]].sort_values("test_season")
        if len(sub) != 3:
            print(f"[warn] {feat}: expected 3 folds, got {len(sub)}")
        row: dict = {"feature": feat}
        for _, r in sub.iterrows():
            ts = int(r["test_season"])
            row[f"dBrier_{ts}"] = float(r["brier_improvement"])
        # fill missing seasons
        for ts in (2022, 2023, 2024):
            k = f"dBrier_{ts}"
            if k not in row:
                row[k] = float("nan")
        row["mean_3fold"] = (row["dBrier_2022"] + row["dBrier_2023"] + row["dBrier_2024"]) / 3
        row["min_fold"] = min(row["dBrier_2022"], row["dBrier_2023"], row["dBrier_2024"])
        row["n_folds_pos"] = sum(
            1 for ts in (2022, 2023, 2024) if row[f"dBrier_{ts}"] > 0
        )
        row["n_folds_ge_thresh"] = sum(
            1 for ts in (2022, 2023, 2024) if row[f"dBrier_{ts}"] >= S_THRESH
        )
        wide_rows.append(row)

    wdf = pd.DataFrame(wide_rows).sort_values("feature")
    print("\n=== Wide: dBrier per test_season (all PASS features) ===\n")
    cols = ["feature", "dBrier_2022", "dBrier_2023", "dBrier_2024", "mean_3fold", "n_folds_pos", "n_folds_ge_thresh"]
    print(wdf[cols].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # Aggregates: mean dBrier at each test season across features
    m22 = float(wdf["dBrier_2022"].mean())
    m23 = float(wdf["dBrier_2023"].mean())
    m24 = float(wdf["dBrier_2024"].mean())
    print("\n=== Mean dBrier across PASS features (unweighted by n_test) ===")
    print(f"  2022 test: {m22:+.5f}")
    print(f"  2023 test: {m23:+.5f}")
    print(f"  2024 test: {m24:+.5f}")
    print(f"  (2024 - 2022) mean shift: {m24 - m22:+.5f}")
    print(f"  (2024 - 2023) mean shift: {m24 - m23:+.5f}")

    n_neg_24 = int((wdf["dBrier_2024"] < 0).sum())
    n_neg_22 = int((wdf["dBrier_2022"] < 0).sum())
    n_neg_23 = int((wdf["dBrier_2023"] < 0).sum())
    print("\n=== Count of features with negative dBrier on that test season ===")
    print(f"  2022: {n_neg_22} / {len(wdf)}")
    print(f"  2023: {n_neg_23} / {len(wdf)}")
    print(f"  2024: {n_neg_24} / {len(wdf)}")

    # Pre-02e only
    w_pre = wdf[wdf["feature"].isin(pre)]
    print("\n=== Pre-02e PASS only: mean dBrier by test season ===")
    print(f"  2022: {w_pre['dBrier_2022'].mean():+.5f}")
    print(f"  2023: {w_pre['dBrier_2023'].mean():+.5f}")
    print(f"  2024: {w_pre['dBrier_2024'].mean():+.5f}")

    w_rz = wdf[wdf["feature"].isin(rz)]
    print("\n=== 02e red-zone PASS only: mean dBrier by test season ===")
    print(w_rz[["feature", "dBrier_2022", "dBrier_2023", "dBrier_2024"]].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    print("\n[done]")


if __name__ == "__main__":
    main()
