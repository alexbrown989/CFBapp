"""Inspect feature_validation.csv to verify imputation_value sign for the seconds_since feature."""
import csv
from pathlib import Path

CSV = Path(r"C:\Users\Alexander\Documents\CFB\CFBapp\research\results\feature_validation.csv")

with open(CSV, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print("Rows for seconds_since_last_dog_explosive_play:")
print(f"  {'train_window':<14} {'test':<5} {'imputation_value':<20} {'brier_imp':<12} {'passed_stability'}")
for row in rows:
    if row["feature"] == "seconds_since_last_dog_explosive_play":
        print(
            f"  {row['train_window']:<14} {row['test_season']:<5} "
            f"{row['imputation_value']:<20} {row['brier_improvement']:<12} {row['passed_stability']}"
        )

print()
print("Rows with non-empty imputation_value (any feature):")
for row in rows:
    iv = (row.get("imputation_value") or "").strip()
    if iv:
        print(
            f"  {row['feature']:<44} {row['train_window']:<14} {row['test_season']:<5} "
            f"imputation_value={iv!r}"
        )

print()
print("Full record for seconds_since_... first window:")
for row in rows:
    if (
        row["feature"] == "seconds_since_last_dog_explosive_play"
        and row["test_season"] == "2022"
    ):
        for k, v in row.items():
            print(f"  {k:<28} = {v!r}")
        break
