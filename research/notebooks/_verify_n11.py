"""Validate the N11 notebook scaffold and, when present, deliverables."""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pandas as pd

NB_PATH = pathlib.Path(__file__).resolve().parent / "11_top25_favorite_stratification.ipynb"
REPO_ROOT = NB_PATH.parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "results"

raw = NB_PATH.read_text(encoding="utf-8")
nb = json.loads(raw)

assert nb["nbformat"] == 4
assert nb["nbformat_minor"] == 5

cells = nb["cells"]
ids = [cell["id"] for cell in cells]
dups = [cid for cid in set(ids) if ids.count(cid) > 1]
assert not dups, f"duplicate cell ids: {dups}"

bad: list[tuple[str, str]] = []
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    assert cell.get("execution_count") in (None, 0)
    assert cell.get("outputs") == []
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as exc:
        bad.append((cell["id"], f"{exc.msg} at line {exc.lineno}"))
if bad:
    for cid, msg in bad:
        print(f"{cid}: {msg}")
    sys.exit(1)

all_text = raw
all_code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")

for marker in [
    "Hypothesis A",
    "Hypothesis B",
    "fav_ap_rank_at_trigger",
    "ranking_bucket",
    "AP Top 25",
    "candidate_live_watch",
    "inverse_hypothesis_sanity_check",
    "favorite_final_win",
    "deficit_erased",
    "pregame_no_vig_implied_prob",
    "n11_ranking_stratification.parquet",
    "n11_analysis_results.json",
    "n11_summary_report.md",
]:
    assert marker in all_text, f"N11 marker missing: {marker!r}"

for forbidden in [
    "httpx.get",
    "requests.get",
    "urllib.request",
    "LogisticRegression(",
    "RandomForest",
    "XGBClassifier",
    "force_refresh=True",
]:
    assert forbidden not in all_code, f"forbidden N11 code path present: {forbidden!r}"

for required in [
    "load_rank_maps",
    "ap_rank_for_team",
    "rank_bucket",
    "bootstrap_cluster_mean",
    "bootstrap_cluster_roi",
    "group_cell_summary",
    "hypothesis_a_separations",
    "hypothesis_b_differentials",
    "inverse_methodology_warning",
    "candidate_live_watch_cells",
]:
    assert required in all_code, f"N11 code marker missing: {required!r}"

ranking_path = RESULTS_DIR / "n11_ranking_stratification.parquet"
analysis_path = RESULTS_DIR / "n11_analysis_results.json"
summary_path = RESULTS_DIR / "n11_summary_report.md"

if any(p.exists() for p in [ranking_path, analysis_path, summary_path]):
    for path in [ranking_path, analysis_path, summary_path]:
        assert path.exists(), f"missing N11 deliverable: {path}"

    ranking = pd.read_parquet(ranking_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(ranking) == 11_412, f"expected 11,412 N11 rows, got {len(ranking):,}"
    key = ["game_id", "fav_deficit", "trigger_sequence"]
    assert ranking[key].duplicated().sum() == 0
    required_cols = [
        "game_id",
        "trigger_play_id",
        "season",
        "week",
        "fav_deficit",
        "time_bucket",
        "trigger_sequence",
        "fav_ap_rank_at_trigger",
        "ap_poll_week_used",
        "ranking_bucket",
        "fluke_bucket",
        "spread_bucket",
        "favorite_final_win",
        "deficit_erased",
        "pregame_raw_implied_prob",
        "pregame_no_vig_implied_prob",
        "decimal_odds_best_available",
        "is_synthetic_fallback_price",
    ]
    for col in required_cols:
        assert col in ranking.columns, f"missing N11 ranking column: {col}"
    assert set(ranking["ranking_bucket"]) == {"top_5", "top_10", "top_25", "unranked"}
    counts = ranking["ranking_bucket"].value_counts().to_dict()
    underpowered = {k: int(v) for k, v in counts.items() if v < 200}
    assert not underpowered, f"ranking buckets underpowered: {underpowered}"
    assert ranking["pregame_no_vig_implied_prob"].between(0, 1).all()
    assert ranking["decimal_odds_best_available"].gt(1).all()

    for key_name in [
        "methodology",
        "ranking_sanity_diagnostics",
        "ranking_single_dimension",
        "matched_ranking_deficit_time_spread_cells",
        "hypothesis_a",
        "hypothesis_b",
        "candidate_live_watch_cells",
        "inverse_hypothesis_sanity_check",
        "inverse_methodology_warning",
    ]:
        assert key_name in analysis, f"missing N11 analysis key: {key_name}"
    assert analysis["methodology"]["models_trained"] == 0
    assert analysis["methodology"]["api_call_count_for_rankings"] == 10
    assert analysis["inverse_methodology_warning"] is False

    sanity = analysis["ranking_sanity_diagnostics"]
    spread = sanity["mean_pregame_spread"]
    assert spread["top_5"] < spread["top_10"] < spread["top_25"] < spread["unranked"]
    fav_rating = sanity["mean_fav_pregame_rating"]
    assert fav_rating["top_5"] > fav_rating["top_25"] > fav_rating["unranked"]
    assert "Hypothesis A" in summary
    assert "Hypothesis B" in summary
    assert "live-watch" in summary
    print(f"[ok] N11 deliverables verified rows={len(ranking):,}, bucket_counts={counts}")
else:
    print("[ok] N11 notebook scaffold verified; deliverables not present yet")

print(f"[ok] N11 notebook structure valid. cells={len(cells)}")
print(f"file: {NB_PATH} ({NB_PATH.stat().st_size:,} bytes)")
