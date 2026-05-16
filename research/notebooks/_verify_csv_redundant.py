"""One-shot verification of feature_validation.csv post-redundancy tag."""
from pathlib import Path
import pandas as pd

csv = Path("research/results/feature_validation.csv")
md = Path("research/results/feature_validation.schema.md")
print(f"CSV : {csv.stat().st_size:,} bytes")
print(f"MD  : {md.stat().st_size:,} bytes")

df = pd.read_csv(csv)
df["redundant_with"] = df["redundant_with"].fillna("")
non_red = df[df["redundant_with"] == ""]
red = df[df["redundant_with"] != ""]
print(f"\nCSV shape: {df.shape}")
print(f"columns: {list(df.columns)}")
print(f"\ncanonical rows (redundant_with == ''): {len(non_red)}")
print(f"canonical features: {sorted(non_red['feature'].unique())}")
print(f"\nredundant rows: {len(red)}")
print(f"redundant features: {sorted(red['feature'].unique())}")
print("\nredundancy mapping:")
for feat, sub in red.groupby("feature"):
    target = sub["redundant_with"].iloc[0]
    print(f"  {feat:<26} -> {target}")
