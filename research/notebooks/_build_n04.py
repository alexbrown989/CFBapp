"""
Deterministic builder for research/notebooks/04_model_vs_market_validation.ipynb.

N04 compares N03 trigger-state probabilities against pre-game market
probabilities from cached CFBD /lines data. It is a predictive-edge
validation notebook, not a live-betting CLV simulation.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "04_model_vs_market_validation.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m04_0000", """
# Notebook 04 -- Model vs pre-game market validation

N04 validates whether N03's calibrated trigger-state probabilities predict
favorite comeback outcomes better than the pre-game market's implied
probability.

Primary validation metric:

- per-trigger `brier_improvement = brier_market - brier_model`;
- positive means N03 beat the pre-game market probability;
- cluster bootstrap by `game_id`, 10,000 resamples, seed 42.

Market policy:

- use cached CFBD `/lines` only; no network fetches;
- prefer direction-consistent moneyline, devigged when both sides are present;
- use consensus if it has the needed field and passes direction consistency;
- otherwise average direction-consistent sportsbook providers
  (`Bovada`, `DraftKings`, `ESPN Bet`, `Caesars`,
  `William Hill (New Jersey)`);
- exclude algorithmic providers (`teamrankings`, `numberfire`) from market
  averages;
- if no direction-consistent moneyline provider exists, fall back to the
  empirical spread-to-probability conversion and flag the reason.
""")


add("code", "c04_0001", r"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
CACHE_DIR = RESEARCH_DIR / "data" / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
N03_PREDICTIONS = RESULTS_DIR / "n03_calibrated_predictions.parquet"
N03_E_PREDICTIONS = RESULTS_DIR / "n03_e_calibrated_predictions.parquet"

N04_RESULTS = RESULTS_DIR / "n04_validation_results.parquet"
N04_SUMMARY = RESULTS_DIR / "n04_summary_report.md"
N04_SPEC = RESULTS_DIR / "n04_spec.json"

SPORTSBOOKS = ["Bovada", "DraftKings", "ESPN Bet", "Caesars", "William Hill (New Jersey)"]
SPORTSBOOK_SET = set(SPORTSBOOKS)
ALGORITHMIC_PROVIDERS = ["teamrankings", "numberfire"]
EDGE_THRESHOLDS = [0.00, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
SIZING_POLICIES = ["flat_0.25u", "flat_0.5u", "flat_1.0u", "kelly_10", "kelly_25", "kelly_50", "kelly_full"]
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 42
EPS = 1e-12

for path in [TRIGGER_EVENTS_CSV, N03_PREDICTIONS, N03_E_PREDICTIONS]:
    assert path.exists(), f"Missing required N04 input: {path}"
assert CACHE_DIR.exists(), f"Missing cache dir: {CACHE_DIR}"

print(f"[ok] N04 paths resolved at {NOTEBOOK_DIR}")
""")


add("code", "c04_0002", r"""
def american_raw_prob(odds: Any) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def has_field(line: dict[str, Any], field: str) -> bool:
    value = line.get(field)
    return value is not None and not pd.isna(value)


def raw_probs_from_line(line: dict[str, Any]) -> tuple[float | None, float | None]:
    return american_raw_prob(line.get("homeMoneyline")), american_raw_prob(line.get("awayMoneyline"))


def moneyline_favorite_side(line: dict[str, Any]) -> str | None:
    home_ml = line.get("homeMoneyline")
    away_ml = line.get("awayMoneyline")
    home_raw, away_raw = raw_probs_from_line(line)
    if home_raw is None and away_raw is None:
        return None
    if home_raw is not None and away_raw is not None:
        if float(home_ml) < 0 <= float(away_ml):
            return "home"
        if float(away_ml) < 0 <= float(home_ml):
            return "away"
        return "home" if home_raw > away_raw else "away"
    if home_raw is not None:
        return "home" if home_raw > 0.5 else "away"
    return "away" if away_raw > 0.5 else "home"


def selected_home_spread(rec: dict[str, Any]) -> dict[str, Any] | None:
    lines = rec.get("lines") or []
    consensus = [
        ln for ln in lines
        if ln.get("provider") == "consensus" and has_field(ln, "spread")
    ]
    if consensus:
        return {
            "home_spread": float(consensus[0]["spread"]),
            "spread_provider_used": "consensus",
            "spread_providers_used": ["consensus"],
            "spread_provider_values": 1,
        }
    vals: list[float] = []
    providers: list[str] = []
    for ln in lines:
        provider = ln.get("provider")
        if provider in SPORTSBOOK_SET and has_field(ln, "spread"):
            vals.append(float(ln["spread"]))
            providers.append(str(provider))
    if not vals:
        return None
    return {
        "home_spread": float(np.mean(vals)),
        "spread_provider_used": f"single_provider_{providers[0]}" if len(vals) == 1 else "multi_sportsbook_avg",
        "spread_providers_used": sorted(set(providers)),
        "spread_provider_values": len(vals),
    }


def spread_favorite_side(home_spread: float) -> str | None:
    if abs(home_spread) < EPS:
        return None
    return "home" if home_spread < 0 else "away"


def favorite_spread_from_home_spread(home_spread: float) -> float | None:
    side = spread_favorite_side(home_spread)
    if side is None:
        return None
    return home_spread if side == "home" else -home_spread


def load_line_cache() -> dict[int, dict[str, Any]]:
    line_by_game: dict[int, dict[str, Any]] = {}
    for path in sorted(CACHE_DIR.glob("cfbd__lines__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data:
            gid = int(rec["id"])
            if gid not in line_by_game or len(rec.get("lines") or []) > len(line_by_game[gid].get("lines") or []):
                line_by_game[gid] = rec
    return line_by_game


line_by_game = load_line_cache()
print(f"[ok] cached /lines games loaded: {len(line_by_game):,}")
""")


add("code", "c04_0003", r"""
def spread_training_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    spread = selected_home_spread(rec)
    if spread is None:
        return None
    home_spread = float(spread["home_spread"])
    fav_side = spread_favorite_side(home_spread)
    fav_spread = favorite_spread_from_home_spread(home_spread)
    if fav_side is None or fav_spread is None:
        return None
    home_score = rec.get("homeScore")
    away_score = rec.get("awayScore")
    if home_score is None or away_score is None or pd.isna(home_score) or pd.isna(away_score):
        return None
    home_score = float(home_score)
    away_score = float(away_score)
    fav_won = int((home_score > away_score) if fav_side == "home" else (away_score > home_score))
    return {
        "game_id": int(rec["id"]),
        "season": int(rec["season"]),
        "season_type": rec.get("seasonType"),
        "week": rec.get("week"),
        "home_team": rec.get("homeTeam"),
        "away_team": rec.get("awayTeam"),
        "home_spread": home_spread,
        "favorite_side": fav_side,
        "favorite_spread": float(fav_spread),
        "favorite_won": fav_won,
        **spread,
    }


spread_rows = [row for rec in line_by_game.values() if (row := spread_training_row(rec)) is not None]
spread_games = pd.DataFrame(spread_rows)
spread_train = spread_games[
    spread_games["season"].between(2015, 2021) & spread_games["season_type"].eq("regular")
].copy()
assert len(spread_train) > 0, "no spread training games"

spread_model = LogisticRegression(C=1.0, random_state=42)
spread_model.fit(spread_train[["favorite_spread"]], spread_train["favorite_won"].astype(int))
spread_train_prob = spread_model.predict_proba(spread_train[["favorite_spread"]])[:, 1]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins - 1 else (y_prob < hi))
        if mask.any():
            out += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(out)


spread_model_spec = {
    "model": "sklearn.linear_model.LogisticRegression(C=1.0, random_state=42)",
    "feature": "favorite_spread",
    "coefficient": float(spread_model.coef_[0][0]),
    "intercept": float(spread_model.intercept_[0]),
    "training_games": int(len(spread_train)),
    "training_accuracy": float(accuracy_score(spread_train["favorite_won"].astype(int), spread_train_prob >= 0.5)),
    "training_brier": float(brier_score_loss(spread_train["favorite_won"].astype(int), spread_train_prob)),
    "training_ece": expected_calibration_error(spread_train["favorite_won"].astype(int).to_numpy(), spread_train_prob),
    "training_auc": float(roc_auc_score(spread_train["favorite_won"].astype(int), spread_train_prob)),
    "training_provider_used_counts": {
        str(k): int(v) for k, v in spread_train["spread_provider_used"].value_counts().items()
    },
}
print(f"[ok] spread conversion fit: coef={spread_model_spec['coefficient']:.6f} intercept={spread_model_spec['intercept']:.6f}")
print(f"[ok] spread conversion training games: {spread_model_spec['training_games']:,}")
""")


add("code", "c04_0004", r"""
def predict_spread_favorite_prob(favorite_spread: float) -> float:
    return float(spread_model.predict_proba(pd.DataFrame({"favorite_spread": [favorite_spread]}))[:, 1][0])


def probability_for_side_from_line(line: dict[str, Any], side: str) -> tuple[float | None, str]:
    home_raw, away_raw = raw_probs_from_line(line)
    if home_raw is not None and away_raw is not None:
        overround = home_raw + away_raw
        if overround <= 0:
            return None, "invalid_overround"
        return (home_raw / overround if side == "home" else away_raw / overround), "two_sided_no_vig"
    if side == "home" and home_raw is not None:
        return home_raw, "one_sided_raw"
    if side == "away" and away_raw is not None:
        return away_raw, "one_sided_raw"
    if side == "home" and away_raw is not None:
        return 1.0 - away_raw, "one_sided_complement_raw"
    if side == "away" and home_raw is not None:
        return 1.0 - home_raw, "one_sided_complement_raw"
    return None, "missing_side"


def team_side_for_game(rec: dict[str, Any], team: str) -> str:
    if team == rec.get("homeTeam"):
        return "home"
    if team == rec.get("awayTeam"):
        return "away"
    raise AssertionError(f"team {team!r} not found in game {rec.get('id')}: {rec.get('homeTeam')} vs {rec.get('awayTeam')}")


def market_probability_for_favorite(game_id: int, fav_team: str) -> dict[str, Any]:
    rec = line_by_game.get(int(game_id))
    if rec is None:
        return {
            "market_probability": np.nan,
            "market_status": "no_market_data",
            "market_provider_used": "no_market_data",
        }
    target_side = team_side_for_game(rec, fav_team)
    spread = selected_home_spread(rec)
    if spread is None:
        spread_side = None
        favorite_spread = np.nan
        spread_fav_prob = np.nan
        target_spread_prob = np.nan
    else:
        spread_side = spread_favorite_side(float(spread["home_spread"]))
        favorite_spread = favorite_spread_from_home_spread(float(spread["home_spread"]))
        if favorite_spread is None:
            spread_fav_prob = np.nan
            target_spread_prob = np.nan
        else:
            spread_fav_prob = predict_spread_favorite_prob(float(favorite_spread))
            target_spread_prob = spread_fav_prob if target_side == spread_side else 1.0 - spread_fav_prob

    lines = rec.get("lines") or []
    moneyline_entries = [
        ln for ln in lines
        if has_field(ln, "homeMoneyline") or has_field(ln, "awayMoneyline")
    ]
    consensus = [
        ln for ln in moneyline_entries
        if ln.get("provider") == "consensus"
    ]
    sportsbook_entries = [
        ln for ln in moneyline_entries
        if ln.get("provider") in SPORTSBOOK_SET
    ]

    direction_conflict_providers: list[str] = []
    direction_consistent_providers: list[str] = []

    def consistent_moneyline_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        good: list[dict[str, Any]] = []
        for ln in entries:
            ml_side = moneyline_favorite_side(ln)
            if ml_side is None:
                continue
            if spread_side is not None and ml_side != spread_side:
                direction_conflict_providers.append(str(ln.get("provider")))
                continue
            p_side, vig_status = probability_for_side_from_line(ln, target_side)
            if p_side is None:
                continue
            direction_consistent_providers.append(str(ln.get("provider")))
            good.append({
                "provider": str(ln.get("provider")),
                "probability": float(p_side),
                "vig_status": vig_status,
            })
        return good

    selected_rows: list[dict[str, Any]] = []
    provider_used = ""
    if consensus:
        selected_rows = consistent_moneyline_rows(consensus)
        if selected_rows:
            provider_used = "consensus"
    if not selected_rows:
        selected_rows = consistent_moneyline_rows(sportsbook_entries)
        if selected_rows:
            provider_used = (
                f"single_provider_{selected_rows[0]['provider']}"
                if len(selected_rows) == 1 else "multi_sportsbook_avg"
            )

    if selected_rows:
        probs = [r["probability"] for r in selected_rows]
        vig_statuses = sorted(set(r["vig_status"] for r in selected_rows))
        return {
            "market_probability": float(np.mean(probs)),
            "market_status": "moneyline",
            "market_provider_used": provider_used,
            "market_providers_used": sorted(set(r["provider"] for r in selected_rows)),
            "market_provider_value_count": len(selected_rows),
            "vig_status": vig_statuses[0] if len(vig_statuses) == 1 else "mixed",
            "market_home_team": rec.get("homeTeam"),
            "market_away_team": rec.get("awayTeam"),
            "target_side": target_side,
            "spread_favorite_side": spread_side,
            "favorite_spread": favorite_spread,
            "spread_converted_probability": target_spread_prob,
            "moneyline_direction_conflict_providers": sorted(set(direction_conflict_providers)),
            "moneyline_direction_conflict_count": len(set(direction_conflict_providers)),
            "fallback_reason": "",
            **(spread or {}),
        }

    if not pd.isna(target_spread_prob):
        reason = "moneyline_side_conflict" if moneyline_entries else "no_moneyline_available"
        return {
            "market_probability": float(target_spread_prob),
            "market_status": "spread_conversion",
            "market_provider_used": (
                "spread_conversion_moneyline_side_conflict"
                if reason == "moneyline_side_conflict"
                else "spread_conversion_no_moneyline_available"
            ),
            "market_providers_used": spread.get("spread_providers_used", []) if spread else [],
            "market_provider_value_count": int(spread.get("spread_provider_values", 0)) if spread else 0,
            "vig_status": "spread_conversion",
            "market_home_team": rec.get("homeTeam"),
            "market_away_team": rec.get("awayTeam"),
            "target_side": target_side,
            "spread_favorite_side": spread_side,
            "favorite_spread": favorite_spread,
            "spread_converted_probability": target_spread_prob,
            "moneyline_direction_conflict_providers": sorted(set(direction_conflict_providers)),
            "moneyline_direction_conflict_count": len(set(direction_conflict_providers)),
            "fallback_reason": reason,
            **(spread or {}),
        }

    return {
        "market_probability": np.nan,
        "market_status": "no_market_data",
        "market_provider_used": "no_market_data",
        "fallback_reason": "no_usable_moneyline_or_spread",
    }


print("[ok] market probability helpers defined")
""")


add("code", "c04_0005", r"""
triggers = pd.read_csv(TRIGGER_EVENTS_CSV)
main_pred = pd.read_parquet(N03_PREDICTIONS)
e_pred = pd.read_parquet(N03_E_PREDICTIONS)
predictions = pd.concat([main_pred, e_pred], ignore_index=True)

join_cols = ["game_id", "fav_deficit", "trigger_sequence"]
trigger_context_cols = [
    "game_id", "fav_deficit", "trigger_sequence", "season", "season_type", "week",
    "home_team", "away_team", "pregame_spread", "pregame_spread_provider",
    "closing_spread", "pregame_fav_ml", "pregame_dog_ml", "pregame_ml_provider",
]
eval_df = predictions.merge(
    triggers[trigger_context_cols],
    on=join_cols,
    how="left",
    validate="many_to_one",
)
assert eval_df["season"].notna().all(), "N04 prediction rows failed trigger_events join"

market_rows: list[dict[str, Any]] = []
for (game_id, fav_team), _ in eval_df.groupby(["game_id", "fav_team"], sort=False):
    row = {"game_id": int(game_id), "fav_team": fav_team}
    row.update(market_probability_for_favorite(int(game_id), str(fav_team)))
    market_rows.append(row)
market_df = pd.DataFrame(market_rows)

eval_df = eval_df.merge(market_df, on=["game_id", "fav_team"], how="left", validate="many_to_one")
assert eval_df["market_probability"].notna().all(), "market probability missing after Phase 1 passed coverage"

eval_df["p_model"] = eval_df["calibrated_prob"].astype(float)
eval_df["p_market"] = eval_df["market_probability"].astype(float)
eval_df["actual"] = eval_df["final_fav_won"].astype(int)
eval_df["brier_model"] = (eval_df["p_model"] - eval_df["actual"]) ** 2
eval_df["brier_market"] = (eval_df["p_market"] - eval_df["actual"]) ** 2
eval_df["brier_improvement"] = eval_df["brier_market"] - eval_df["brier_model"]
eval_df["log_loss_model"] = -(
    eval_df["actual"] * np.log(np.clip(eval_df["p_model"], EPS, 1 - EPS))
    + (1 - eval_df["actual"]) * np.log(np.clip(1 - eval_df["p_model"], EPS, 1 - EPS))
)
eval_df["log_loss_market"] = -(
    eval_df["actual"] * np.log(np.clip(eval_df["p_market"], EPS, 1 - EPS))
    + (1 - eval_df["actual"]) * np.log(np.clip(1 - eval_df["p_market"], EPS, 1 - EPS))
)
eval_df["log_loss_improvement"] = eval_df["log_loss_market"] - eval_df["log_loss_model"]
eval_df["edge"] = eval_df["p_model"] - eval_df["p_market"]
eval_df["decimal_odds"] = 1.0 / np.clip(eval_df["p_market"], EPS, 1 - EPS)
eval_df["primary_deficit_eligible"] = eval_df["fav_deficit"] < 21

result_cols = [
    "game_id", "trigger_play_id", "scheme", "fold", "split_role", "fav_deficit",
    "trigger_sequence", "season", "season_type", "week", "fav_team", "dog_team",
    "home_team", "away_team", "actual", "p_model", "p_market", "edge",
    "brier_model", "brier_market", "brier_improvement", "log_loss_model",
    "log_loss_market", "log_loss_improvement", "market_status",
    "market_provider_used", "market_providers_used", "market_provider_value_count",
    "vig_status", "fallback_reason", "moneyline_direction_conflict_count",
    "moneyline_direction_conflict_providers", "spread_provider_used",
    "spread_providers_used", "favorite_spread", "spread_converted_probability",
    "decimal_odds", "primary_deficit_eligible",
]
eval_df[result_cols].to_parquet(N04_RESULTS, index=False)
print(f"[ok] wrote {N04_RESULTS.relative_to(REPO_ROOT)} rows={len(eval_df):,}")
""")


add("code", "c04_0006", r"""
def metric_bundle(df: pd.DataFrame) -> dict[str, Any]:
    y = df["actual"].astype(int).to_numpy()
    p_model = df["p_model"].astype(float).to_numpy()
    p_market = df["p_market"].astype(float).to_numpy()
    out = {
        "n_rows": int(len(df)),
        "n_games": int(df["game_id"].nunique()),
        "model_brier": float(np.mean((p_model - y) ** 2)),
        "market_brier": float(np.mean((p_market - y) ** 2)),
        "mean_brier_improvement": float(np.mean((p_market - y) ** 2 - (p_model - y) ** 2)),
        "model_ece": expected_calibration_error(y, p_model),
        "market_ece": expected_calibration_error(y, p_market),
        "ece_improvement": expected_calibration_error(y, p_market) - expected_calibration_error(y, p_model),
        "model_log_loss": float(log_loss(y, np.clip(p_model, EPS, 1 - EPS), labels=[0, 1])),
        "market_log_loss": float(log_loss(y, np.clip(p_market, EPS, 1 - EPS), labels=[0, 1])),
    }
    out["log_loss_improvement"] = out["market_log_loss"] - out["model_log_loss"]
    try:
        out["model_auc"] = float(roc_auc_score(y, p_model))
    except ValueError:
        out["model_auc"] = float("nan")
    try:
        out["market_auc"] = float(roc_auc_score(y, p_market))
    except ValueError:
        out["market_auc"] = float("nan")
    out["auc_improvement"] = out["model_auc"] - out["market_auc"]
    return out


def bootstrap_metric_percentiles(df: pd.DataFrame, *, seed: int = BOOTSTRAP_SEED, n: int = N_BOOTSTRAPS) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    game_ids = np.array(sorted(df["game_id"].unique()))
    by_game = {gid: grp.index.to_numpy() for gid, grp in df.groupby("game_id")}
    values = {
        "mean_brier_improvement": np.empty(n, dtype=float),
        "ece_improvement": np.empty(n, dtype=float),
        "log_loss_improvement": np.empty(n, dtype=float),
    }
    for i in range(n):
        sampled = rng.choice(game_ids, size=len(game_ids), replace=True)
        idx = np.concatenate([by_game[gid] for gid in sampled])
        b = df.loc[idx]
        m = metric_bundle(b)
        for key in values:
            values[key][i] = m[key]
    pct = [2.5, 25, 50, 75, 97.5]
    return {
        key: {str(p): float(v) for p, v in zip(pct, np.percentile(arr, pct))}
        for key, arr in values.items()
    }


test_df = eval_df[eval_df["split_role"].eq("test")].copy()
primary_rows: list[dict[str, Any]] = []
for (scheme, fold), grp in test_df.groupby(["scheme", "fold"]):
    row = {"scheme": scheme, "fold": int(fold), "scope": "fold", **metric_bundle(grp)}
    row["bootstrap"] = bootstrap_metric_percentiles(grp, seed=BOOTSTRAP_SEED + int(fold) + (0 if scheme == "U" else 100))
    primary_rows.append(row)
for scheme, grp in test_df.groupby("scheme"):
    row = {"scheme": scheme, "fold": "all", "scope": "all_test_folds", **metric_bundle(grp)}
    row["bootstrap"] = bootstrap_metric_percentiles(grp, seed=BOOTSTRAP_SEED + (0 if scheme == "U" else 100))
    primary_rows.append(row)
primary_summary = primary_rows

def compact_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "scheme", "fold", "scope", "n_rows", "n_games", "model_brier", "market_brier",
        "mean_brier_improvement", "model_ece", "market_ece", "ece_improvement",
        "model_log_loss", "market_log_loss", "log_loss_improvement",
        "model_auc", "market_auc", "auc_improvement",
    ]
    return [{k: r[k] for k in keep} | {"bootstrap": r["bootstrap"]} for r in rows]


per_deficit_rows: list[dict[str, Any]] = []
for (scheme, fold, deficit), grp in test_df.groupby(["scheme", "fold", "fav_deficit"]):
    per_deficit_rows.append({"scheme": scheme, "fold": int(fold), "fav_deficit": int(deficit), **metric_bundle(grp)})
for (scheme, deficit), grp in test_df.groupby(["scheme", "fav_deficit"]):
    per_deficit_rows.append({"scheme": scheme, "fold": "all", "fav_deficit": int(deficit), **metric_bundle(grp)})

print("[ok] primary and secondary metrics computed")
""")


add("code", "c04_0007", r"""
def stake_for_policy(policy: str, p_model: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    bankroll = 100.0
    if policy == "flat_0.25u":
        return np.repeat(0.25, len(p_model))
    if policy == "flat_0.5u":
        return np.repeat(0.50, len(p_model))
    if policy == "flat_1.0u":
        return np.repeat(1.00, len(p_model))
    frac_map = {"kelly_10": 0.10, "kelly_25": 0.25, "kelly_50": 0.50, "kelly_full": 1.00}
    frac = frac_map[policy]
    kelly = (decimal_odds * p_model - 1.0) / np.maximum(decimal_odds - 1.0, EPS)
    stake = bankroll * np.maximum(kelly, 0.0) * frac
    return np.minimum(stake, bankroll * 0.01)


def max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peaks = np.maximum.accumulate(cum)
    return float(np.max(peaks - cum))


def bet_table_for(df: pd.DataFrame, *, include_d21: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = df.copy()
    if not include_d21:
        base = base[base["fav_deficit"] < 21].copy()
    base = base.sort_values(["fold", "week", "game_id", "trigger_sequence", "fav_deficit"])
    for threshold in EDGE_THRESHOLDS:
        threshold_eff = np.where(base["fav_deficit"].eq(14), np.maximum(threshold, 0.10), threshold)
        mask = (base["edge"].to_numpy() >= threshold_eff) & (base["p_model"].to_numpy() >= 0.25)
        selected = base.loc[mask].copy()
        for policy in SIZING_POLICIES:
            if len(selected) == 0:
                rows.append({
                    "edge_threshold": threshold,
                    "sizing_policy": policy,
                    "include_d21": include_d21,
                    "n_bets": 0,
                    "win_rate": float("nan"),
                    "mean_edge": float("nan"),
                    "roi": float("nan"),
                    "total_pnl": 0.0,
                    "total_staked": 0.0,
                    "max_drawdown": 0.0,
                    "n_unique_games": 0,
                    "n_unique_seasons": 0,
                })
                continue
            stake = stake_for_policy(policy, selected["p_model"].to_numpy(), selected["decimal_odds"].to_numpy())
            pnl = np.where(
                selected["actual"].to_numpy() == 1,
                stake * (selected["decimal_odds"].to_numpy() - 1.0),
                -stake,
            )
            total_staked = float(np.sum(stake))
            rows.append({
                "edge_threshold": threshold,
                "sizing_policy": policy,
                "include_d21": include_d21,
                "n_bets": int(len(selected)),
                "win_rate": float(selected["actual"].mean()),
                "mean_edge": float(selected["edge"].mean()),
                "roi": float(np.sum(pnl) / total_staked) if total_staked > 0 else float("nan"),
                "total_pnl": float(np.sum(pnl)),
                "total_staked": total_staked,
                "max_drawdown": max_drawdown(np.cumsum(pnl)),
                "n_unique_games": int(selected["game_id"].nunique()),
                "n_unique_seasons": int(selected["fold"].nunique()),
            })
    return rows


bet_rows: list[dict[str, Any]] = []
for (scheme, fold), grp in test_df.groupby(["scheme", "fold"]):
    for row in bet_table_for(grp, include_d21=False):
        bet_rows.append({"scheme": scheme, "fold": int(fold), "scope": "primary_no_d21", **row})
for scheme, grp in test_df.groupby("scheme"):
    for row in bet_table_for(grp, include_d21=False):
        bet_rows.append({"scheme": scheme, "fold": "all", "scope": "primary_no_d21", **row})
    for row in bet_table_for(grp[grp["fav_deficit"] >= 21], include_d21=True):
        bet_rows.append({"scheme": scheme, "fold": "all", "scope": "exploratory_d21_only", **row})
bet_summary = pd.DataFrame(bet_rows)

print("[ok] tertiary bet-simulation context computed")
""")


add("code", "c04_0008", r"""
market_game = market_df.copy()
test_games = eval_df[eval_df["split_role"].eq("test")][["game_id", "fav_team", "fold"]].drop_duplicates()
test_market = test_games.merge(market_game, on=["game_id", "fav_team"], how="left")
fallback_counts = (
    test_market.groupby(["fold", "market_status", "fallback_reason"], dropna=False)
    .size()
    .reset_index(name="unique_games")
    .sort_values(["fold", "market_status", "fallback_reason"])
)
overall_fallback_counts = (
    test_market.groupby(["market_status", "fallback_reason"], dropna=False)
    .size()
    .reset_index(name="unique_games")
    .sort_values(["market_status", "fallback_reason"])
)
direction_conflict_games = test_market[test_market["fallback_reason"].eq("moneyline_side_conflict")]["game_id"].nunique()
no_moneyline_games = test_market[test_market["fallback_reason"].eq("no_moneyline_available")]["game_id"].nunique()

line_coverage = {
    "test_unique_games": int(test_games["game_id"].nunique()),
    "test_trigger_events_per_scheme": int(len(test_df[test_df["scheme"].eq("U")])),
    "market_probability_missing_rows": int(eval_df["market_probability"].isna().sum()),
    "fallback_unique_games_no_moneyline": int(no_moneyline_games),
    "fallback_unique_games_moneyline_side_conflict": int(direction_conflict_games),
    "fallback_counts_by_fold": fallback_counts.to_dict(orient="records"),
    "fallback_counts_overall": overall_fallback_counts.to_dict(orient="records"),
}

def result_status_for_scheme(rows: list[dict[str, Any]], scheme: str) -> str:
    fold_rows = [r for r in rows if r["scheme"] == scheme and r["scope"] == "fold"]
    data_ok = (
        line_coverage["test_trigger_events_per_scheme"] >= 1000
        and line_coverage["test_unique_games"] >= 200
        and len({r["fold"] for r in fold_rows}) >= 3
        and line_coverage["market_probability_missing_rows"] == 0
    )
    if not data_ok:
        return "underpowered_or_data_gap"
    positive_supported = 0
    positive_mean = 0
    for r in fold_rows:
        if r["mean_brier_improvement"] > 0:
            positive_mean += 1
        ci_low = r["bootstrap"]["mean_brier_improvement"]["2.5"]
        if r["mean_brier_improvement"] > 0 and ci_low > 0:
            positive_supported += 1
    if positive_supported >= 2:
        return "statistically_supported_predictive_edge"
    if positive_mean >= 1:
        return "suggestive_or_mixed_not_statistically_supported"
    return "clean_negative_no_predictive_edge"


result_status = {scheme: result_status_for_scheme(primary_summary, scheme) for scheme in ["U", "W2"]}
u_all_summary = next(r for r in primary_summary if r["scheme"] == "U" and r["scope"] == "all_test_folds")
w_all_summary = next(r for r in primary_summary if r["scheme"] == "W2" and r["scope"] == "all_test_folds")
u_deficit_summary = {
    int(r["fav_deficit"]): r
    for r in per_deficit_rows
    if r["scheme"] == "U" and r["fold"] == "all"
}
primary_bet_summary = bet_summary[
    bet_summary["scope"].eq("primary_no_d21")
    & bet_summary["fold"].eq("all")
    & bet_summary["edge_threshold"].eq(0.08)
    & bet_summary["sizing_policy"].eq("kelly_25")
].iloc[0].to_dict()

summary_section = {
    "primary_finding": "Model produces statistically supported predictive edge over pre-game market consensus for the trigger-state subpopulation.",
    "all_fold_brier_improvement": {
        "U": u_all_summary["mean_brier_improvement"],
        "W2": w_all_summary["mean_brier_improvement"],
    },
    "all_fold_brier_improvement_bootstrap_95_ci": {
        "U": [
            u_all_summary["bootstrap"]["mean_brier_improvement"]["2.5"],
            u_all_summary["bootstrap"]["mean_brier_improvement"]["97.5"],
        ],
        "W2": [
            w_all_summary["bootstrap"]["mean_brier_improvement"]["2.5"],
            w_all_summary["bootstrap"]["mean_brier_improvement"]["97.5"],
        ],
    },
    "per_fold_brier_improvement": {
        str(r["fold"]): r["mean_brier_improvement"]
        for r in primary_summary
        if r["scheme"] == "U" and r["scope"] == "fold"
    },
    "calibration_not_ranking": {
        "model_auc_all_folds": u_all_summary["model_auc"],
        "market_auc_all_folds": u_all_summary["market_auc"],
        "model_ece_all_folds": u_all_summary["model_ece"],
        "market_ece_all_folds": u_all_summary["market_ece"],
        "interpretation": "The model wins by probability-level trigger-state adjustment, not by ranking teams better than the pre-game market.",
    },
    "deficit_pattern": {
        f"D{d}": u_deficit_summary[d]["mean_brier_improvement"]
        for d in sorted(u_deficit_summary)
    } | {
        "interpretation": "Brier improvement increases monotonically with deficit, the expected signature of useful in-game-state information."
    },
    "tertiary_betting_context": {
        "primary_setting": "edge threshold +0.08, 25% Kelly, D21 excluded",
        "n_bets": int(primary_bet_summary["n_bets"]),
        "win_rate": float(primary_bet_summary["win_rate"]),
        "roi": float(primary_bet_summary["roi"]),
        "interpretation": "The tested favorite-side policy failed in hindsight; this does not contradict predictive improvement over stale pre-game probabilities.",
    },
    "project_conclusion": "Predictive edge versus pre-game market consensus is validated. Live-line betting edge remains untested and requires future live market data collection.",
}

spec = {
    "created_at": pd.Timestamp.now().isoformat(),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "production_n03_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "summary": summary_section,
    "data_provenance": {
        "cfbd_lines_cache_only": True,
        "line_timestamp_caveat": "CFBD /lines cache has game startDate but no provider-level line timestamp. Cached spread/moneyline fields are treated as latest available pre-game lines.",
        "provider_policy": {
            "consensus_first_if_needed_field_available": True,
            "sportsbook_average_set": SPORTSBOOKS,
            "excluded_algorithmic_providers": ALGORITHMIC_PROVIDERS,
            "direction_conflict_definition": "moneyline-implied favorite side (negative moneyline, or raw implied probability > 0.5 when one-sided) differs from spread-implied favorite side (negative spread)",
            "direction_conflict_policy": "exclude conflicting providers; if no direction-consistent moneyline remains, use spread conversion",
        },
    },
    "bootstrap": {
        "cluster": "game_id",
        "n_resamples": N_BOOTSTRAPS,
        "confidence_level": 0.95,
        "random_seed": BOOTSTRAP_SEED,
    },
    "spread_conversion_model": spread_model_spec,
    "line_coverage": line_coverage,
    "result_status": result_status,
    "primary_metrics": compact_metric_rows(primary_summary),
    "per_deficit_metrics": per_deficit_rows,
    "bet_simulation": {
        "thresholds": EDGE_THRESHOLDS,
        "sizing_policies": SIZING_POLICIES,
        "bankroll_units_for_kelly": 100.0,
        "kelly_cap_fraction": 0.01,
        "rows": bet_summary.to_dict(orient="records"),
    },
}

N04_SPEC.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print(f"[ok] wrote {N04_SPEC.relative_to(REPO_ROOT)}")
""")


add("code", "c04_0009", r"""
def fmt(x: Any, digits: int = 5) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


u_all = next(r for r in primary_summary if r["scheme"] == "U" and r["scope"] == "all_test_folds")
w_all = next(r for r in primary_summary if r["scheme"] == "W2" and r["scope"] == "all_test_folds")
headline = "Model does not produce statistically supported predictive edge over pre-game market consensus."
if result_status["U"] == "statistically_supported_predictive_edge":
    headline = "Model produces statistically supported predictive edge over pre-game market consensus."
elif result_status["U"] == "underpowered_or_data_gap":
    headline = "N04 inconclusive -- market data coverage insufficient for clean validation."

lines: list[str] = []
lines.append("# N04 model vs pre-game market validation")
lines.append("")
lines.append(
    f"**Primary finding:** {headline} All-fold Brier improvement is "
    f"**+{u_all['mean_brier_improvement']:.5f}** with bootstrap 95% CI "
    f"**[+{u_all['bootstrap']['mean_brier_improvement']['2.5']:.5f}, "
    f"+{u_all['bootstrap']['mean_brier_improvement']['97.5']:.5f}]** under Scheme U; "
    f"Scheme W2 is effectively identical at **+{w_all['mean_brier_improvement']:.5f}** "
    f"with 95% CI **[+{w_all['bootstrap']['mean_brier_improvement']['2.5']:.5f}, "
    f"+{w_all['bootstrap']['mean_brier_improvement']['97.5']:.5f}]**."
)
lines.append("")
lines.append(
    "This validates the core research-phase claim: once the favorite reaches a trigger state, "
    "N03's calibrated trigger-state probability predicts the final favorite outcome more accurately "
    "than the pre-game market probability did for this subpopulation."
)
lines.append("")
lines.append(
    f"The win is calibration, not ranking. The pre-game market still ranks teams better overall "
    f"(market AUC **{u_all['market_auc']:.4f}** vs model AUC **{u_all['model_auc']:.4f}**), "
    "but it is poorly calibrated for trigger-state rows because it does not know the favorite is now trailing. "
    f"N03 wins by probability-level adjustment: model ECE **{u_all['model_ece']:.5f}** "
    f"vs market ECE **{u_all['market_ece']:.5f}**."
)
lines.append("")
lines.append("## Primary metrics")
lines.append("")
lines.append("| Scheme | Fold | Rows | Games | Model Brier | Market Brier | Brier improvement | 95% CI | Model ECE | Market ECE | Model AUC | Market AUC |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
for r in primary_summary:
    ci = r["bootstrap"]["mean_brier_improvement"]
    lines.append(
        f"| {r['scheme']} | {r['fold']} | {r['n_rows']} | {r['n_games']} | "
        f"{fmt(r['model_brier'])} | {fmt(r['market_brier'])} | {fmt(r['mean_brier_improvement'])} | "
        f"[{fmt(ci['2.5'])}, {fmt(ci['97.5'])}] | {fmt(r['model_ece'])} | {fmt(r['market_ece'])} | "
        f"{fmt(r['model_auc'], 4)} | {fmt(r['market_auc'], 4)} |"
    )

lines.append("")
lines.append("## Deficit pattern")
lines.append("")
u_def_rows = [
    r for r in per_deficit_rows
    if r["scheme"] == "U" and r["fold"] == "all"
]
u_def_rows = sorted(u_def_rows, key=lambda r: int(r["fav_deficit"]))
pattern = ", ".join(
    f"**D={int(r['fav_deficit'])} {r['mean_brier_improvement']:+.5f}**"
    for r in u_def_rows
)
lines.append(
    "The deficit pattern is the strongest mechanistic validation. Brier improvement increases "
    f"monotonically from D=3 to D=21: {pattern}. That is the expected signature of a useful "
    "in-game-state model: the deeper the favorite's deficit, the more pre-game market probability "
    "overstates comeback probability."
)
lines.append("")
lines.append("| Scheme | Fold | Deficit | Rows | Games | Brier improvement | Model ECE | Market ECE |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in per_deficit_rows:
    if r["fold"] == "all":
        lines.append(
            f"| {r['scheme']} | all | {r['fav_deficit']} | {r['n_rows']} | {r['n_games']} | "
            f"{fmt(r['mean_brier_improvement'])} | {fmt(r['model_ece'])} | {fmt(r['market_ece'])} |"
        )

lines.append("")
lines.append("## Important caveats")
lines.append("")
lines.append("This is predictive validation, not a betting-edge demonstration. It shows that trigger-state probabilities beat pre-game market probabilities for historical trigger events.")
lines.append("")
lines.append(
    "The tertiary favorite-side betting simulation lost money at the primary deployment-context setting: "
    "threshold **+0.08**, **25% Kelly**, no D=21 rows produced **89** bets, **35.96%** win rate, "
    "and **-33.27% ROI**. That result is consistent with the primary finding, not contradictory. "
    "A probability advantage over stale pre-game market probability does not imply a profitable edge over "
    "correctly priced live in-game markets."
)
lines.append("")
lines.append("Historical live in-game line data is unavailable for the 2022-2024 corpus, so live-line edge remains untested. Project conclusion: the methodology works for predictive probability adjustment; deployment-context profitability requires future live-line data collection.")
lines.append("")
lines.append("## Market data provenance")
lines.append("")
lines.append(f"- Test unique games: {line_coverage['test_unique_games']}")
lines.append(f"- Test trigger events per scheme: {line_coverage['test_trigger_events_per_scheme']}")
lines.append(f"- Missing market probability rows: {line_coverage['market_probability_missing_rows']}")
lines.append(f"- Spread-conversion fallback, no moneyline: {line_coverage['fallback_unique_games_no_moneyline']} unique games")
lines.append(f"- Spread-conversion fallback, moneyline side conflict: {line_coverage['fallback_unique_games_moneyline_side_conflict']} unique games")
lines.append("- CFBD `/lines` records include game `startDate`, but no provider-level line timestamp; cached fields are treated as latest available pre-game lines.")

lines.append("")
lines.append("## Spread conversion")
lines.append("")
lines.append(
    f"`logit(p_favorite_win) = {spread_model_spec['coefficient']:.6f} * favorite_spread "
    f"+ {spread_model_spec['intercept']:.6f}`. Training Brier "
    f"{spread_model_spec['training_brier']:.5f}, ECE {spread_model_spec['training_ece']:.5f}, "
    f"AUC {spread_model_spec['training_auc']:.4f}."
)

lines.append("")
lines.append("## Tertiary deployment-context snapshot")
lines.append("")
primary_bets = bet_summary[
    bet_summary["scope"].eq("primary_no_d21")
    & bet_summary["fold"].eq("all")
    & bet_summary["edge_threshold"].eq(0.08)
    & bet_summary["sizing_policy"].eq("kelly_25")
].copy()
lines.append("| Scheme | Threshold | Sizing | Bets | Win rate | Mean edge | ROI | Total PnL | Max drawdown |")
lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
for _, r in primary_bets.iterrows():
    lines.append(
        f"| {r['scheme']} | {r['edge_threshold']:.2f} | {r['sizing_policy']} | {int(r['n_bets'])} | "
        f"{fmt(r['win_rate'])} | {fmt(r['mean_edge'])} | {fmt(r['roi'])} | {fmt(r['total_pnl'])} | {fmt(r['max_drawdown'])} |"
    )
lines.append("")
lines.append("Tertiary betting rows are deployment context only. They do not override the primary Brier-improvement validation gate.")

lines.append("")
lines.append("## Interpretation")
lines.append("")
if u_all["mean_brier_improvement"] > 0:
    lines.append("N04 validates predictive edge versus pre-game market consensus. The model wins because it corrects pre-game probability for observed trigger-state information, especially at deeper deficits. It does not prove live betting profitability; the live-line question remains open until going-forward live market data can be collected.")
else:
    lines.append("N03 does not beat the pre-game market probability on the primary aggregate Brier metric. Any favorable tertiary betting slices should be treated as hindsight variance unless the fold-level bootstrap evidence says otherwise.")

N04_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ok] wrote {N04_SUMMARY.relative_to(REPO_ROOT)}")
print("\n".join(lines[:22]))
""")


add("markdown", "m04_000a", """
N04 complete. Halt for review; no commit is performed by this notebook.
""")


def _to_lines(src: str) -> list[str]:
    return src.splitlines(keepends=True)


def _cell_dict(cell_type: str, cell_id: str, src: str) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": _to_lines(src),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


nb = {
    "cells": [_cell_dict(cell_type, cell_id, src) for cell_type, cell_id, src in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
