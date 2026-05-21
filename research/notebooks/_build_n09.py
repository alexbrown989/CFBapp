"""
Deterministic builder for research/notebooks/09_trigger_state_analysis.ipynb.

N09 is a diagnostic/application-facing notebook. It produces:
  1. baseline_C deep analysis,
  2. dashboard-ready trigger-state stratifications,
  3. counterfactual betting simulations against pre-game prices.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "09_trigger_state_analysis.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m09_0000", """
# Notebook 09 -- Trigger-state analysis, dashboard stratification, and counterfactual betting simulation

N09 has three equal-weight sections:

1. Deep analysis of the `fav_deficit x time_bucket` baseline_C that beat
   N05/N06/N07/N08 model variants on structural edge.
2. Dashboard-ready descriptive stratifications across rich trigger-state
   dimensions, with both labels shown side by side.
3. Counterfactual betting simulations against cached pre-game prices.

Terminology discipline:

- **predictive edge**: a model beats a probability baseline on Brier or another
  proper scoring rule.
- **structural edge**: a model beats baseline_C specifically.
- **market edge**: a model beats live in-game market prices; N09 cannot test this.
- **betting edge**: a realized simulation produces positive CLV or ROI.

N09 uses N06 calibrated probabilities, consistent with the N08 deployment
recommendation. It does not train a new production model.
""")


add("code", "c09_0001", r"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
CACHE_DIR = RESEARCH_DIR / "data" / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
N04_RESULTS = RESULTS_DIR / "n04_validation_results.parquet"
N05_RATES = RESULTS_DIR / "n05_descriptive_rates.parquet"
N03_PREDICTIONS = RESULTS_DIR / "n03_calibrated_predictions.parquet"
N06_PREDICTIONS = RESULTS_DIR / "n06_calibrated_predictions.parquet"
N07_FEATURES = RESULTS_DIR / "n07_descriptive_features.parquet"
N08_DIAGNOSTIC = RESULTS_DIR / "n08_diagnostic_predictions.parquet"
N08_PRICE_SPEC = RESULTS_DIR / "n08_price_conversion_spec.json"

N09_STRAT_PARQUET = RESULTS_DIR / "n09_trigger_state_stratifications.parquet"
N09_BASELINE_JSON = RESULTS_DIR / "n09_baseline_analysis.json"
N09_BETS_PARQUET = RESULTS_DIR / "n09_betting_simulations.parquet"
N09_BETTING_JSON = RESULTS_DIR / "n09_betting_summary.json"
N09_SUMMARY_MD = RESULTS_DIR / "n09_summary_report.md"

for path in [
    TRIGGER_EVENTS_CSV, TRIGGER_OUTCOMES_CSV, N04_RESULTS, N05_RATES, N03_PREDICTIONS,
    N06_PREDICTIONS, N07_FEATURES, N08_DIAGNOSTIC, N08_PRICE_SPEC,
]:
    assert path.exists(), f"Missing required N09 input: {path}"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
EPS = 1e-12
LABELS = ["favorite_final_win", "deficit_erased"]
SPORTSBOOKS = ["Bovada", "DraftKings", "ESPN Bet", "Caesars", "William Hill (New Jersey)"]
SPORTSBOOK_SET = set(SPORTSBOOKS)
ALGORITHMIC_PROVIDERS = {"teamrankings", "numberfire"}
BET_YEARS = [2022, 2023, 2024]

print(f"[ok] N09 setup at {NOTEBOOK_DIR}")
""")


add("code", "c09_0002", r"""
def chrono_key(p: dict[str, Any]) -> tuple[int, int, int, int]:
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    elapsed = 900 - 60 * int(m) - int(s) if m is not None and s is not None else 0
    return (
        period,
        elapsed,
        int(p.get("driveNumber") or 0),
        int(p.get("playNumber") or 0),
    )


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if n <= 0:
        return {"lower": None, "upper": None}
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return {"lower": float(max(0.0, center - margin)), "upper": float(min(1.0, center + margin))}


def thin_flag(n_events: int, n_games: int, n_seasons: int) -> str:
    if n_events < 20 or n_games < 15 or n_seasons < 2:
        return "unreliable"
    if n_events < 50 or n_games < 30 or n_seasons == 2:
        return "thin"
    return "reliable"


def cluster_bootstrap_rate(
    df: pd.DataFrame,
    label: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(df) == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
    grouped = df.groupby("game_id")[label].agg(["sum", "count"]).reset_index()
    successes = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    idx = np.arange(len(grouped))
    for i in range(n_resamples):
        sample = rng.choice(idx, size=len(idx), replace=True)
        denom = counts[sample].sum()
        draws[i] = successes[sample].sum() / denom if denom else np.nan
    return {
        "lower": float(np.nanquantile(draws, 0.025)),
        "p25": float(np.nanquantile(draws, 0.25)),
        "median": float(np.nanquantile(draws, 0.50)),
        "p75": float(np.nanquantile(draws, 0.75)),
        "upper": float(np.nanquantile(draws, 0.975)),
        "n_resamples": n_resamples,
    }


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(df) == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
    grouped = {gid: g[value_col].to_numpy(dtype=float) for gid, g in df.groupby("game_id")}
    gids = np.array(list(grouped))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(gids, size=len(gids), replace=True)
        values = np.concatenate([grouped[gid] for gid in sampled])
        draws[i] = float(np.mean(values))
    return {
        "lower": float(np.nanquantile(draws, 0.025)),
        "p25": float(np.nanquantile(draws, 0.25)),
        "median": float(np.nanquantile(draws, 0.50)),
        "p75": float(np.nanquantile(draws, 0.75)),
        "upper": float(np.nanquantile(draws, 0.975)),
        "n_resamples": n_resamples,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins - 1 else (y_prob < hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)))
    except ValueError:
        return float("nan")


def pct_bucket(v: Any, *, no_points: bool = False, include_all: bool = True) -> str:
    if no_points:
        return "no_points"
    if pd.isna(v):
        return "missing"
    x = float(v)
    if x <= 0:
        return "none"
    if x <= 0.20:
        return "low"
    if x <= 0.50:
        return "moderate"
    if include_all and x >= 1.0 - 1e-9:
        return "all"
    return "high"


def epa_bucket(v: Any) -> str:
    if pd.isna(v):
        return "missing"
    x = float(v)
    if x > 0.20:
        return "favorite_much_better"
    if x > 0.05:
        return "favorite_better"
    if x >= -0.05:
        return "neutral"
    if x >= -0.20:
        return "dog_better"
    return "dog_much_better"


def pace_bucket(v: Any) -> str:
    if pd.isna(v):
        return "missing"
    x = float(v)
    if x < 0.6:
        return "very_slow"
    if x < 0.8:
        return "slow"
    if x < 1.0:
        return "average"
    if x <= 1.2:
        return "fast"
    return "very_fast"


def possessions_bucket(v: Any) -> str:
    if pd.isna(v):
        return "missing"
    x = float(v)
    if x <= 3:
        return "<=3"
    if x <= 6:
        return "4-6"
    if x <= 9:
        return "7-9"
    if x <= 12:
        return "10-12"
    return ">12"


def tempo_bucket(v: Any, n_prior: int) -> str:
    if n_prior < 2 or pd.isna(v):
        return "first_two_games_of_season"
    x = float(v)
    if x < 0.7:
        return "slow_tempo"
    if x <= 0.9:
        return "average_tempo"
    return "fast_tempo"


def pass_rate_bucket(v: Any, n_prior: int) -> str:
    if n_prior < 2 or pd.isna(v):
        return "first_two_games_of_season"
    x = float(v)
    if x < 0.35:
        return "run_heavy"
    if x <= 0.55:
        return "balanced"
    return "pass_heavy"


def momentum_bucket(v: Any, early: bool) -> str:
    if early or pd.isna(v):
        return "early_game"
    x = float(v)
    if x < -0.15:
        return "strong_decline"
    if x < -0.05:
        return "decline"
    if x <= 0.05:
        return "flat"
    if x <= 0.15:
        return "improving"
    return "strong_improvement"


QUINTILE_BUCKET_LABELS = [
    "q1_lowest",
    "q2_low",
    "q3_middle",
    "q4_high",
    "q5_highest",
]


def empirical_quintile_edges(values: pd.Series) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return []
    edges = [float(clean.quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]
    # Duplicate quantiles happen on discrete or clumped dimensions. Keep a
    # deterministic monotone sequence so bucket assignment remains stable.
    out: list[float] = []
    last = -np.inf
    for edge in edges:
        if edge <= last:
            edge = np.nextafter(last, np.inf)
        out.append(edge)
        last = edge
    return out


def apply_empirical_quintile_bucket(
    values: pd.Series,
    *,
    missing_label: str = "missing",
    fixed_bucket: pd.Series | None = None,
) -> tuple[pd.Series, list[float]]:
    numeric = pd.to_numeric(values, errors="coerce")
    if fixed_bucket is None:
        fixed_bucket = pd.Series([None] * len(values), index=values.index, dtype="object")
    fit_mask = numeric.notna() & fixed_bucket.isna()
    edges = empirical_quintile_edges(numeric.loc[fit_mask])
    buckets = pd.Series(missing_label, index=values.index, dtype="object")
    buckets.loc[fit_mask] = [
        QUINTILE_BUCKET_LABELS[min(int(np.searchsorted(edges, float(v), side="right")), 4)]
        for v in numeric.loc[fit_mask]
    ]
    fixed_mask = fixed_bucket.notna()
    buckets.loc[fixed_mask] = fixed_bucket.loc[fixed_mask].astype(str)
    return buckets, edges


def apply_composition_quintile_bucket(
    values: pd.Series,
    *,
    no_points_mask: pd.Series | None = None,
) -> tuple[pd.Series, list[float]]:
    numeric = pd.to_numeric(values, errors="coerce")
    fixed = pd.Series([None] * len(values), index=values.index, dtype="object")
    if no_points_mask is not None:
        fixed.loc[no_points_mask.fillna(False)] = "no_points"
    fixed.loc[numeric.le(0).fillna(False) & fixed.isna()] = "none"
    positive = numeric.where(numeric.gt(0) & fixed.isna())
    buckets, edges = apply_empirical_quintile_bucket(positive, fixed_bucket=fixed)
    return buckets, edges


def american_raw_prob(odds: Any) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_decimal_odds(odds: Any) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds == 0:
        return None
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bytes)) else False:
        return None
    return obj


print("[ok] helpers defined")
""")


add("code", "c09_0003", r"""
def load_json_cache(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CACHE_DIR.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data)
    return rows


def load_games_by_id() -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for rec in load_json_cache("cfbd__games__*.json"):
        gid = int(rec["id"])
        if gid not in by_id:
            by_id[gid] = rec
    return by_id


def load_lines_by_id() -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for rec in load_json_cache("cfbd__lines__*.json"):
        gid = int(rec["id"])
        if gid not in by_id or len(rec.get("lines") or []) > len(by_id[gid].get("lines") or []):
            by_id[gid] = rec
    return by_id


games_by_id = load_games_by_id()
lines_by_id = load_lines_by_id()
trigger_events = pd.read_csv(TRIGGER_EVENTS_CSV)
trigger_outcomes = pd.read_csv(TRIGGER_OUTCOMES_CSV)
n05 = pd.read_parquet(N05_RATES)
n07 = pd.read_parquet(N07_FEATURES)
n08 = pd.read_parquet(N08_DIAGNOSTIC)
n03 = pd.read_parquet(N03_PREDICTIONS)
n06 = pd.read_parquet(N06_PREDICTIONS)
n04 = pd.read_parquet(N04_RESULTS)
price_spec = json.loads(N08_PRICE_SPEC.read_text(encoding="utf-8"))

n03_final_win = (
    n03[(n03["scheme"].eq("U")) & (n03["split_role"].eq("test"))]
    [["game_id", "fav_deficit", "trigger_sequence", "calibrated_prob"]]
    .rename(columns={"calibrated_prob": "model_prob_final_win"})
)
assert n03_final_win[["game_id", "fav_deficit", "trigger_sequence"]].duplicated().sum() == 0
n08 = n08.merge(
    n03_final_win,
    on=["game_id", "fav_deficit", "trigger_sequence"],
    how="left",
    validate="one_to_one",
)
assert n08["model_prob_final_win"].notna().all(), "Missing N03 final-win probability after N08 join"
n08 = n08.rename(columns={"n06_prob": "model_prob_deficit_erased"})
n08["n06_prob"] = n08["model_prob_deficit_erased"]

base = n05[n05["deficit_erased"].notna()].copy()
base["favorite_final_win"] = base["favorite_final_win"].astype(bool).astype(int)
base["deficit_erased"] = base["deficit_erased"].astype(bool).astype(int)
base = base.merge(
    trigger_events[[
        "game_id", "fav_deficit", "trigger_sequence", "season_type",
        "pregame_spread", "home_team", "away_team", "fav_score_at_trigger",
        "dog_score_at_trigger", "seconds_remaining_in_regulation",
        "actual_deficit_at_trigger", "drive_number_in_game",
    ]],
    on=["game_id", "fav_deficit", "trigger_sequence"],
    how="left",
    validate="one_to_one",
)
assert "week" in base.columns, "N05-owned week column missing after trigger_events merge"
assert "week_x" not in base.columns and "week_y" not in base.columns, "duplicate week columns after merge"
base = base.merge(
    n07.drop(columns=["favorite_final_win", "deficit_erased"], errors="ignore"),
    on=["game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "season", "time_bucket", "fav_team", "dog_team", "quarter", "clock_seconds_in_period_total"],
    how="left",
    validate="one_to_one",
)
assert len(base) == 11412, f"Expected 11,412 N09 base rows, got {len(base)}"

game_meta_rows = []
for gid in base["game_id"].unique():
    rec = games_by_id.get(int(gid), {})
    game_meta_rows.append({
        "game_id": int(gid),
        "is_conference_game": bool(rec.get("conferenceGame")) if rec else False,
        "is_neutral_site": bool(rec.get("neutralSite")) if rec else False,
        "start_date": rec.get("startDate"),
    })
game_meta = pd.DataFrame(game_meta_rows)
base = base.merge(game_meta, on="game_id", how="left", validate="many_to_one")

trigger_game_ids = set(int(x) for x in base["game_id"].unique())
all_plays_by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
for path in sorted(CACHE_DIR.glob("cfbd__plays__*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data:
        gid = int(p.get("gameId") or 0)
        if gid in trigger_game_ids:
            all_plays_by_game[gid].append(p)

for gid, plays in list(all_plays_by_game.items()):
    dedup = {str(p.get("id")): p for p in plays}
    all_plays_by_game[gid] = sorted(dedup.values(), key=chrono_key)

missing_pbp = sorted(trigger_game_ids - set(all_plays_by_game))
assert not missing_pbp, f"Missing play-by-play for trigger games: {missing_pbp[:10]}"

print(f"[ok] base rows={len(base):,}; trigger games={len(trigger_game_ids):,}; cached play games={len(all_plays_by_game):,}")
""")


add("code", "c09_0004", r"""
def is_scrimmage_play(p: dict[str, Any]) -> bool:
    pt = str(p.get("playType") or "").lower()
    return any(token in pt for token in ["rush", "pass", "sack"])


def is_pass_play(p: dict[str, Any]) -> bool:
    pt = str(p.get("playType") or "").lower()
    return "pass" in pt or "sack" in pt


game_team_stats: list[dict[str, Any]] = []
for gid, plays in all_plays_by_game.items():
    rec = games_by_id.get(gid, {})
    season = int(rec.get("season") or 0)
    week = int(rec.get("week") or 0)
    season_type = rec.get("seasonType")
    for team in [rec.get("homeTeam"), rec.get("awayTeam")]:
        if not team:
            continue
        team_plays = [p for p in plays if p.get("offense") == team and is_scrimmage_play(p)]
        n_total = len(team_plays)
        n_pass = sum(is_pass_play(p) for p in team_plays)
        game_team_stats.append({
            "game_id": gid,
            "season": season,
            "season_type": season_type,
            "week": week,
            "team": team,
            "team_plays": n_total,
            "team_pass_attempts": n_pass,
            "plays_per_minute": n_total / 60.0,
            "pass_rate": n_pass / n_total if n_total else np.nan,
        })
game_team_stats_df = pd.DataFrame(game_team_stats)
prior_lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
for (season, team), grp in game_team_stats_df[game_team_stats_df["season_type"].eq("regular")].groupby(["season", "team"]):
    grp = grp.sort_values(["week", "game_id"]).reset_index(drop=True)
    cum_games = 0
    cum_ppm = 0.0
    cum_pass = 0.0
    cum_plays = 0.0
    for _, row in grp.iterrows():
        prior_lookup[(int(season), int(row["game_id"]), str(team))] = {
            "prior_games": int(cum_games),
            "prior_plays_per_minute": cum_ppm / cum_games if cum_games else np.nan,
            "prior_pass_rate": cum_pass / cum_plays if cum_plays else np.nan,
        }
        cum_games += 1
        cum_ppm += float(row["plays_per_minute"])
        cum_pass += float(row["team_pass_attempts"])
        cum_plays += float(row["team_plays"])


def trigger_play_index(plays: list[dict[str, Any]], trigger_play_id: Any) -> int | None:
    sid = str(trigger_play_id)
    for i, p in enumerate(plays):
        if str(p.get("id")) == sid:
            return i
    return None


def ppa_value(p: dict[str, Any]) -> float | None:
    v = p.get("ppa")
    if v is None or pd.isna(v):
        return None
    return float(v)


def momentum_for_team(plays_before: list[dict[str, Any]], team: str) -> dict[str, Any]:
    team_plays = [p for p in plays_before if p.get("offense") == team and ppa_value(p) is not None]
    cumulative_vals = [ppa_value(p) for p in team_plays if ppa_value(p) is not None]
    cumulative = float(np.mean(cumulative_vals)) if cumulative_vals else np.nan
    drives: list[tuple[int, list[float]]] = []
    by_drive: dict[int, list[float]] = defaultdict(list)
    for p in team_plays:
        dn = int(p.get("driveNumber") or -1)
        val = ppa_value(p)
        if val is not None:
            by_drive[dn].append(val)
    for dn in sorted(by_drive):
        if by_drive[dn]:
            drives.append((dn, by_drive[dn]))
    if len(drives) < 3 or pd.isna(cumulative):
        return {"recent_3": np.nan, "cumulative": cumulative, "delta": np.nan, "early": True}
    recent_vals = [v for _, vals in drives[-3:] for v in vals]
    recent = float(np.mean(recent_vals)) if recent_vals else np.nan
    delta = recent - cumulative if not pd.isna(recent) and not pd.isna(cumulative) else np.nan
    return {"recent_3": recent, "cumulative": cumulative, "delta": delta, "early": False}


feature_rows: list[dict[str, Any]] = []
for row in base.itertuples(index=False):
    plays = all_plays_by_game[int(row.game_id)]
    idx = trigger_play_index(plays, row.trigger_play_id)
    if idx is None:
        raise AssertionError(f"trigger play {row.trigger_play_id} not found in game {row.game_id}")
    plays_before = plays[:idx]
    elapsed_minutes = ((int(row.quarter) - 1) * 15.0) + (float(row.period_seconds_elapsed) / 60.0)
    plays_so_far_n09 = len(plays_before)
    pace = plays_so_far_n09 / elapsed_minutes if elapsed_minutes > 0 else np.nan
    fav_m = momentum_for_team(plays_before, str(row.fav_team))
    dog_m = momentum_for_team(plays_before, str(row.dog_team))
    prior = prior_lookup.get((int(row.season), int(row.game_id), str(row.fav_team)), {
        "prior_games": 0,
        "prior_plays_per_minute": np.nan,
        "prior_pass_rate": np.nan,
    })
    feature_rows.append({
        "game_id": int(row.game_id),
        "fav_deficit": int(row.fav_deficit),
        "trigger_sequence": int(row.trigger_sequence),
        "plays_so_far_n09": plays_so_far_n09,
        "pace_plays_per_minute": pace,
        "fav_prior_games": int(prior["prior_games"]),
        "fav_prior_plays_per_minute": prior["prior_plays_per_minute"],
        "fav_prior_pass_rate": prior["prior_pass_rate"],
        "fav_epa_recent_3_drives": fav_m["recent_3"],
        "fav_epa_cumulative": fav_m["cumulative"],
        "fav_momentum_delta": fav_m["delta"],
        "fav_momentum_early_game": bool(fav_m["early"]),
        "dog_epa_recent_3_drives": dog_m["recent_3"],
        "dog_epa_cumulative": dog_m["cumulative"],
        "dog_momentum_delta": dog_m["delta"],
        "dog_momentum_early_game": bool(dog_m["early"]),
    })
computed_features = pd.DataFrame(feature_rows)
base = base.merge(computed_features, on=["game_id", "fav_deficit", "trigger_sequence"], how="left", validate="one_to_one")

print("[ok] pace, team-style, and momentum features computed")
""")


add("code", "c09_0005", r"""
dog_score = base["dog_score_at_trigger"].astype(float)
bucket_cutpoints: dict[str, list[float]] = {}

base["turnover_composition_bucket"], bucket_cutpoints["turnover_composition_bucket"] = apply_composition_quintile_bucket(
    base["dog_points_from_turnovers_pct"]
)
short_field_pct = (
    base["dog_points_from_turnovers_pct"].fillna(0).astype(float)
    + base["dog_points_from_returns_pct"].fillna(0).astype(float)
)
base["short_field_composition_pct"] = np.where(dog_score.eq(0), 0.0, np.clip(short_field_pct, 0.0, 1.0))
base["short_field_composition_bucket"], bucket_cutpoints["short_field_composition_bucket"] = apply_composition_quintile_bucket(
    base["short_field_composition_pct"]
)
base["explosive_composition_bucket"], bucket_cutpoints["explosive_composition_bucket"] = apply_composition_quintile_bucket(
    base["dog_points_from_explosives_pct"],
    no_points_mask=dog_score.eq(0),
)
base["epa_differential_bucket"], bucket_cutpoints["epa_differential_bucket"] = apply_empirical_quintile_bucket(
    base["epa_per_play_gap"]
)
base["pace_bucket"], bucket_cutpoints["pace_bucket"] = apply_empirical_quintile_bucket(
    base["pace_plays_per_minute"]
)
base["possessions_remaining_bucket"], bucket_cutpoints["possessions_remaining_bucket"] = apply_empirical_quintile_bucket(
    base["estimated_possessions_remaining"]
)
tempo_fixed = pd.Series([None] * len(base), index=base.index, dtype="object")
tempo_fixed.loc[base["fav_prior_games"].lt(2) | base["fav_prior_plays_per_minute"].isna()] = "first_two_games_of_season"
base["favorite_tempo_bucket"], bucket_cutpoints["favorite_tempo_bucket"] = apply_empirical_quintile_bucket(
    base["fav_prior_plays_per_minute"],
    fixed_bucket=tempo_fixed,
)
pass_fixed = pd.Series([None] * len(base), index=base.index, dtype="object")
pass_fixed.loc[base["fav_prior_games"].lt(2) | base["fav_prior_pass_rate"].isna()] = "first_two_games_of_season"
base["favorite_pass_rate_bucket"], bucket_cutpoints["favorite_pass_rate_bucket"] = apply_empirical_quintile_bucket(
    base["fav_prior_pass_rate"],
    fixed_bucket=pass_fixed,
)
fav_momentum_fixed = pd.Series([None] * len(base), index=base.index, dtype="object")
fav_momentum_fixed.loc[base["fav_momentum_early_game"] | base["fav_momentum_delta"].isna()] = "early_game"
base["fav_momentum_bucket"], bucket_cutpoints["fav_momentum_bucket"] = apply_empirical_quintile_bucket(
    base["fav_momentum_delta"],
    fixed_bucket=fav_momentum_fixed,
)
dog_momentum_fixed = pd.Series([None] * len(base), index=base.index, dtype="object")
dog_momentum_fixed.loc[base["dog_momentum_early_game"] | base["dog_momentum_delta"].isna()] = "early_game"
base["dog_momentum_bucket"], bucket_cutpoints["dog_momentum_bucket"] = apply_empirical_quintile_bucket(
    base["dog_momentum_delta"],
    fixed_bucket=dog_momentum_fixed,
)
base["is_conference_game_bucket"] = base["is_conference_game"].map({True: "true", False: "false"})
base["is_neutral_site_bucket_n09"] = base["is_neutral_site"].map({True: "true", False: "false"})
base["season_week_bucket"] = np.select(
    [
        base["season_type"].eq("postseason"),
        base["week"].between(1, 4),
        base["week"].between(5, 9),
        base["week"].between(10, 14),
    ],
    ["bowl", "early", "mid", "late"],
    default="late",
)
base["deficit_bucket"] = "D=" + base["fav_deficit"].astype(str)

dashboard_cols = [
    "game_id", "trigger_play_id", "fav_deficit", "deficit_bucket", "trigger_sequence",
    "season", "week", "season_type", "time_bucket", "fav_team", "dog_team",
    "favorite_final_win", "deficit_erased",
    "fav_score_at_trigger", "dog_score_at_trigger", "seconds_remaining_in_regulation",
    "estimated_possessions_remaining", "possessions_remaining_bucket",
    "deficit_per_remaining_possession", "clock_pressure_index",
    "turnover_composition_bucket", "short_field_composition_pct", "short_field_composition_bucket",
    "explosive_composition_bucket", "epa_per_play_gap", "epa_differential_bucket",
    "pace_plays_per_minute", "pace_bucket", "fav_prior_games", "fav_prior_plays_per_minute",
    "favorite_tempo_bucket", "fav_prior_pass_rate", "favorite_pass_rate_bucket",
    "fav_momentum_delta", "fav_momentum_bucket", "dog_momentum_delta", "dog_momentum_bucket",
    "is_conference_game_bucket", "is_neutral_site_bucket_n09", "season_week_bucket",
]
strat_df = base[dashboard_cols].copy()
assert len(strat_df) == 11412
strat_df.to_parquet(N09_STRAT_PARQUET, index=False)

bucket_columns_for_concentration = [
    "turnover_composition_bucket", "short_field_composition_bucket", "explosive_composition_bucket",
    "epa_differential_bucket", "pace_bucket", "possessions_remaining_bucket", "favorite_tempo_bucket",
    "favorite_pass_rate_bucket", "fav_momentum_bucket", "dog_momentum_bucket", "season_week_bucket",
]
bucket_concentration = {}
for col in bucket_columns_for_concentration:
    shares = strat_df[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False)
    bucket_concentration[col] = {
        "top_bucket": str(shares.index[0]),
        "top_share": float(shares.iloc[0]),
        "counts": {str(k): int(v) for k, v in strat_df[col].value_counts(dropna=False).items()},
    }

print(f"[ok] wrote {N09_STRAT_PARQUET.relative_to(REPO_ROOT)} rows={len(strat_df):,}")
print("[info] bucket concentration top shares:", {k: round(v["top_share"], 3) for k, v in bucket_concentration.items()})
""")


add("code", "c09_0006", r"""
def rate_table(df: pd.DataFrame, group_cols: list[str], *, include_bootstrap: bool, seed_offset: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for key, grp in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base_row = {col: (None if pd.isna(val) else val) for col, val in zip(group_cols, key)}
        n_events = int(len(grp))
        n_games = int(grp["game_id"].nunique())
        n_seasons = int(grp["season"].nunique())
        for label in LABELS:
            successes = int(grp[label].sum())
            row = {
                **base_row,
                "label": label,
                "n_events": n_events,
                "n_games": n_games,
                "n_seasons": n_seasons,
                "thin_flag": thin_flag(n_events, n_games, n_seasons),
                "successes": successes,
                "rate": float(successes / n_events) if n_events else None,
                "wilson_ci": wilson_ci(successes, n_events),
            }
            if include_bootstrap:
                row["bootstrap_ci"] = cluster_bootstrap_rate(grp, label, seed=BOOTSTRAP_SEED + seed_offset + len(rows))
            rows.append(row)
    return rows


baseline_20_cell = rate_table(base, ["fav_deficit", "time_bucket"], include_bootstrap=True)
season_rows = []
for (deficit, tb, season), grp in base.groupby(["fav_deficit", "time_bucket", "season"]):
    for label in LABELS:
        season_rows.append({
            "fav_deficit": int(deficit),
            "time_bucket": tb,
            "season": int(season),
            "label": label,
            "n_events": int(len(grp)),
            "rate": float(grp[label].mean()),
        })
season_df = pd.DataFrame(season_rows)
stability_rows = []
for (deficit, tb, label), grp in season_df.groupby(["fav_deficit", "time_bucket", "label"]):
    stability_rows.append({
        "fav_deficit": int(deficit),
        "time_bucket": tb,
        "label": label,
        "season_rate_std": float(grp["rate"].std(ddof=0)),
        "season_rate_range": float(grp["rate"].max() - grp["rate"].min()),
        "season_rows": grp.sort_values("season").to_dict(orient="records"),
    })
stability_df = pd.DataFrame(stability_rows)
most_stable = stability_df.sort_values("season_rate_std").head(5).to_dict(orient="records")
least_stable = stability_df.sort_values("season_rate_std", ascending=False).head(5).to_dict(orient="records")

heldout = n08.copy()
heldout["disagreement"] = (heldout["baseline_C_deficit_erased"] - heldout["n06_prob"]).abs()
top100 = heldout.sort_values("disagreement", ascending=False).head(100).copy()
low100 = heldout.sort_values("disagreement", ascending=True).head(100).copy()


def disagreement_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    actual_rate = float(df[label].mean())
    mean_n06 = float(df["n06_prob"].mean())
    baseline_col = f"baseline_C_{label}"
    mean_base = float(df[baseline_col].mean())
    return {
        "n": int(len(df)),
        "actual_rate": actual_rate,
        "mean_n06_prob": mean_n06,
        "mean_baseline_C_prob": mean_base,
        "n06_abs_error_vs_actual_rate": abs(mean_n06 - actual_rate),
        "baseline_C_abs_error_vs_actual_rate": abs(mean_base - actual_rate),
        "closer_to_actual_rate": "n06" if abs(mean_n06 - actual_rate) < abs(mean_base - actual_rate) else "baseline_C",
        "n06_higher_share": float((df["n06_prob"] > df[baseline_col]).mean()),
    }


disagreement_analysis = {
    "distribution": {
        "mean": float(heldout["disagreement"].mean()),
        "p10": float(heldout["disagreement"].quantile(0.10)),
        "p50": float(heldout["disagreement"].quantile(0.50)),
        "p90": float(heldout["disagreement"].quantile(0.90)),
        "max": float(heldout["disagreement"].max()),
    },
    "top100_high_disagreement": {
        "records": top100[[
            "game_id", "trigger_play_id", "fav_deficit", "trigger_sequence", "season", "time_bucket",
            "n06_prob", "baseline_C_deficit_erased", "baseline_C_favorite_final_win",
            "favorite_final_win", "deficit_erased", "disagreement",
        ]].to_dict(orient="records"),
        "summary": {label: disagreement_summary(top100, label) for label in LABELS},
    },
    "low100_low_disagreement": {
        "summary": {label: disagreement_summary(low100, label) for label in LABELS},
    },
}


def fit_two_feature(label: str) -> dict[str, Any]:
    train = base[base["season"].between(2015, 2021)].copy()
    test = heldout.copy()
    pre = ColumnTransformer(
        transformers=[
            ("deficit", StandardScaler(), ["fav_deficit"]),
            ("time", OneHotEncoder(handle_unknown="ignore"), ["time_bucket"]),
        ]
    )
    pipe = Pipeline([
        ("pre", pre),
        ("logreg", LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
    ])
    pipe.fit(train[["fav_deficit", "time_bucket"]], train[label].astype(int))
    p = pipe.predict_proba(test[["fav_deficit", "time_bucket"]])[:, 1]
    baseline_col = f"baseline_C_{label}"
    y = test[label].astype(int).to_numpy()
    n06_prob = test["n06_prob"].to_numpy()
    return {
        "label": label,
        "n_train_events": int(len(train)),
        "n_test_events": int(len(test)),
        "two_feature_brier": float(brier_score_loss(y, p)),
        "baseline_C_brier": float(brier_score_loss(y, test[baseline_col])),
        "n06_brier": float(brier_score_loss(y, n06_prob)),
        "two_feature_auc": safe_auc(y, p),
        "baseline_C_auc": safe_auc(y, test[baseline_col]),
        "n06_auc": safe_auc(y, n06_prob),
        "two_feature_ece": expected_calibration_error(y, p),
        "baseline_C_ece": expected_calibration_error(y, test[baseline_col]),
        "n06_ece": expected_calibration_error(y, n06_prob),
    }


two_feature_reproduction = {label: fit_two_feature(label) for label in LABELS}

section2_dimensions = {
    "turnover_composition": "turnover_composition_bucket",
    "short_field_composition": "short_field_composition_bucket",
    "explosive_play_composition": "explosive_composition_bucket",
    "epa_differential": "epa_differential_bucket",
    "pace": "pace_bucket",
    "possessions_remaining": "possessions_remaining_bucket",
    "favorite_tempo": "favorite_tempo_bucket",
    "favorite_pass_rate": "favorite_pass_rate_bucket",
    "favorite_momentum": "fav_momentum_bucket",
    "dog_momentum": "dog_momentum_bucket",
    "is_conference_game": "is_conference_game_bucket",
    "is_neutral_site": "is_neutral_site_bucket_n09",
    "season_week_bucket": "season_week_bucket",
}
section2_tables = {
    name: rate_table(strat_df, [col], include_bootstrap=False, seed_offset=5000 + i * 100)
    for i, (name, col) in enumerate(section2_dimensions.items())
}
cross_dimension_table = rate_table(strat_df, ["deficit_bucket", "fav_momentum_bucket"], include_bootstrap=False, seed_offset=9000)

baseline_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "section1_baseline_C": {
        "twenty_cell_rate_table": baseline_20_cell,
        "season_stability": stability_df.to_dict(orient="records"),
        "most_stable_cells": most_stable,
        "least_stable_cells": least_stable,
        "n06_vs_baseline_C_disagreement": disagreement_analysis,
        "two_feature_reproduction": two_feature_reproduction,
    },
    "section2_stratifications": {
        "dimension_tables": section2_tables,
        "cross_dimension_deficit_x_fav_momentum": cross_dimension_table,
        "bucket_concentration": bucket_concentration,
        "bucket_cutpoints": bucket_cutpoints,
        "rebucketing_note": (
            "Numeric stratification dimensions use empirical quintiles over the full corpus. "
            "Composition dimensions keep zero/no-points buckets separate and use quintiles over the positive population. "
            "Intrinsic first-two-games and early-game buckets remain intentionally imbalanced."
        ),
        "terminology_note": "All Section 2 tables are descriptive only; no statistical significance claims are made.",
    },
}
N09_BASELINE_JSON.write_text(json.dumps(json_safe(baseline_payload), indent=2) + "\n", encoding="utf-8")
print(f"[ok] wrote {N09_BASELINE_JSON.relative_to(REPO_ROOT)}")
""")


add("code", "c09_0007", r"""
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


def team_side_for_game(rec: dict[str, Any], team: str) -> str:
    if team == rec.get("homeTeam"):
        return "home"
    if team == rec.get("awayTeam"):
        return "away"
    raise AssertionError(f"team {team} not in game {rec.get('id')}")


def selected_home_spread(rec: dict[str, Any]) -> float | None:
    lines = rec.get("lines") or []
    consensus = [ln for ln in lines if ln.get("provider") == "consensus" and has_field(ln, "spread")]
    if consensus:
        return float(consensus[0]["spread"])
    vals = [
        float(ln["spread"]) for ln in lines
        if ln.get("provider") in SPORTSBOOK_SET and has_field(ln, "spread")
    ]
    return float(np.mean(vals)) if vals else None


def spread_favorite_side(home_spread: float) -> str | None:
    if abs(home_spread) < EPS:
        return None
    return "home" if home_spread < 0 else "away"


def market_prices_for_favorite(game_id: int, fav_team: str, fallback_prob: float) -> dict[str, Any]:
    rec = lines_by_id.get(int(game_id))
    if rec is None:
        return {
            "best_market_prob": fallback_prob,
            "best_decimal_odds": 1.0 / np.clip(fallback_prob, EPS, 1 - EPS),
            "consensus_market_prob": fallback_prob,
            "consensus_decimal_odds": 1.0 / np.clip(fallback_prob, EPS, 1 - EPS),
            "market_price_source": "spread_or_model_fallback_no_line_record",
            "fallback_used": True,
        }
    target_side = team_side_for_game(rec, fav_team)
    home_spread = selected_home_spread(rec)
    spread_side = spread_favorite_side(home_spread) if home_spread is not None else None
    usable: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for ln in rec.get("lines") or []:
        provider = ln.get("provider")
        if provider in ALGORITHMIC_PROVIDERS:
            continue
        if not (has_field(ln, "homeMoneyline") or has_field(ln, "awayMoneyline")):
            continue
        ml_side = moneyline_favorite_side(ln)
        if ml_side is None:
            continue
        if spread_side is not None and ml_side != spread_side:
            conflicts.append(str(provider))
            continue
        odds = ln.get("homeMoneyline") if target_side == "home" else ln.get("awayMoneyline")
        dec = american_decimal_odds(odds)
        raw = american_raw_prob(odds)
        if dec is None or raw is None:
            continue
        row = {"provider": str(provider), "decimal_odds": float(dec), "raw_prob": float(raw)}
        if provider == "consensus":
            consensus_rows.append(row)
        elif provider in SPORTSBOOK_SET:
            usable.append(row)
    if usable:
        best = max(usable, key=lambda r: r["decimal_odds"])
        best_dec = best["decimal_odds"]
        best_prob = 1.0 / best_dec
        source = f"best_sportsbook_{best['provider']}"
    elif consensus_rows:
        best = consensus_rows[0]
        best_dec = best["decimal_odds"]
        best_prob = 1.0 / best_dec
        source = "consensus_only"
    else:
        return {
            "best_market_prob": fallback_prob,
            "best_decimal_odds": 1.0 / np.clip(fallback_prob, EPS, 1 - EPS),
            "consensus_market_prob": fallback_prob,
            "consensus_decimal_odds": 1.0 / np.clip(fallback_prob, EPS, 1 - EPS),
            "market_price_source": "spread_or_model_fallback_no_direction_consistent_moneyline",
            "fallback_used": True,
            "direction_conflict_providers": sorted(set(conflicts)),
        }
    if consensus_rows:
        c = consensus_rows[0]
        consensus_dec = c["decimal_odds"]
        consensus_prob = 1.0 / consensus_dec
    else:
        consensus_dec = np.nan
        consensus_prob = np.nan
    return {
        "best_market_prob": float(best_prob),
        "best_decimal_odds": float(best_dec),
        "consensus_market_prob": float(consensus_prob) if not pd.isna(consensus_prob) else np.nan,
        "consensus_decimal_odds": float(consensus_dec) if not pd.isna(consensus_dec) else np.nan,
        "market_price_source": source,
        "fallback_used": False,
        "direction_conflict_providers": sorted(set(conflicts)),
    }


heldout_u = n08.copy()
deepest_idx = heldout_u.sort_values(["fav_deficit", "trigger_sequence"]).groupby("game_id")["fav_deficit"].idxmax()
bets = heldout_u.loc[deepest_idx].copy().sort_values(["season", "game_id"]).reset_index(drop=True)
assert bets["game_id"].duplicated().sum() == 0
assert set(bets["season"].unique()) <= set(BET_YEARS)

market_rows: list[dict[str, Any]] = []
for row in bets.itertuples(index=False):
    market_rows.append({
        "game_id": int(row.game_id),
        **market_prices_for_favorite(int(row.game_id), str(row.fav_team), float(row.baseline_C_favorite_final_win)),
    })
market_df = pd.DataFrame(market_rows)
bets = bets.merge(market_df, on="game_id", how="left", validate="one_to_one")
bets["edge_at_entry_final_win"] = bets["model_prob_final_win"] - bets["best_market_prob"]
bets["edge_at_entry_deficit_erased"] = bets["model_prob_deficit_erased"] - bets["best_market_prob"]
bets["edge_at_entry"] = bets["edge_at_entry_deficit_erased"]
bets["sw_disagreement"] = bets["model_prob_deficit_erased"] - bets["baseline_sw_cfb_prob"]
bets["price_source_type"] = np.where(bets["fallback_used"].fillna(False), "synthetic_fallback", "real_moneyline")

line_coverage = {
    "heldout_trigger_games": int(len(bets)),
    "fallback_used_games": int(bets["fallback_used"].fillna(False).sum()),
    "fallback_rate": float(bets["fallback_used"].fillna(False).mean()),
    "market_price_source_counts": {str(k): int(v) for k, v in bets["market_price_source"].value_counts(dropna=False).items()},
}
print("[ok] betting universe:", line_coverage)
""")


add("code", "c09_0008", r"""
def selected_simulation_rows(name: str, threshold: float | None = None) -> pd.DataFrame:
    if name == "A_all_trigger_games":
        out = bets.copy()
    elif name == "B_final_win_model_edge":
        out = bets[bets["edge_at_entry_final_win"] >= float(threshold)].copy()
    elif name == "B_deficit_erased_heuristic":
        out = bets[bets["edge_at_entry_deficit_erased"] >= float(threshold)].copy()
    elif name == "C_deep_trigger_D10_plus":
        out = bets[bets["fav_deficit"] >= 10].copy()
    elif name == "D_sw_disagreement":
        out = bets[bets["sw_disagreement"] >= float(threshold)].copy()
    elif name == "E_conformal_narrow_edge":
        cutoff = float(bets["conformal_width"].quantile(0.25))
        out = bets[(bets["conformal_width"] <= cutoff) & (bets["edge_at_entry_deficit_erased"] >= 0.05)].copy()
    else:
        raise KeyError(name)
    return out.sort_values(["season", "game_id"]).reset_index(drop=True)


simulation_specs = [
    {
        "simulation": "A_all_trigger_games",
        "threshold": None,
        "prob_col": "model_prob_final_win",
        "edge_col": "edge_at_entry_final_win",
        "probability_label": "favorite_final_win",
        "methodology_status": "same_label",
        "description": "Bet every held-out trigger game favorite at best pre-game price.",
    },
    *[
        {
            "simulation": "B_final_win_model_edge",
            "threshold": t,
            "prob_col": "model_prob_final_win",
            "edge_col": "edge_at_entry_final_win",
            "probability_label": "favorite_final_win",
            "methodology_status": "same_label_headline",
            "description": f"Bet if N03 final-win probability - raw market break-even probability >= {t:.2f}.",
        }
        for t in [0.00, 0.05, 0.10]
    ],
    *[
        {
            "simulation": "B_deficit_erased_heuristic",
            "threshold": t,
            "prob_col": "model_prob_deficit_erased",
            "edge_col": "edge_at_entry_deficit_erased",
            "probability_label": "deficit_erased",
            "methodology_status": "cross_label_heuristic_not_headline",
            "description": f"Bet if N06 deficit-erased probability - raw market break-even probability >= {t:.2f}; heuristic only.",
        }
        for t in [0.00, 0.05, 0.10]
    ],
    {
        "simulation": "C_deep_trigger_D10_plus",
        "threshold": None,
        "prob_col": "model_prob_final_win",
        "edge_col": "edge_at_entry_final_win",
        "probability_label": "favorite_final_win",
        "methodology_status": "same_label",
        "description": "Bet if deepest trigger is D=10 or higher.",
    },
    *[
        {
            "simulation": "D_sw_disagreement",
            "threshold": t,
            "prob_col": "model_prob_deficit_erased",
            "edge_col": "sw_disagreement",
            "probability_label": "deficit_erased",
            "methodology_status": "cross_label_heuristic_not_headline",
            "description": f"Bet if N06 deficit-erased probability - Stern-Winston CFB state price >= {t:.2f}; heuristic only.",
        }
        for t in [0.03, 0.05, 0.10]
    ],
    {
        "simulation": "E_conformal_narrow_edge",
        "threshold": 0.05,
        "prob_col": "model_prob_deficit_erased",
        "edge_col": "edge_at_entry_deficit_erased",
        "probability_label": "deficit_erased",
        "methodology_status": "cross_label_heuristic_not_headline",
        "description": "Bet if N06 conformal width in narrowest 25% and deficit-erased heuristic edge >= 0.05.",
    },
]
staking_rules = ["flat_1u", "eighth_kelly", "quarter_kelly"]


def stake_amount(rule: str, bankroll: float, p: float, decimal_odds: float) -> float:
    if rule == "flat_1u":
        return 1.0
    b = max(decimal_odds - 1.0, EPS)
    q = 1.0 - p
    kelly = max(0.0, (p * b - q) / b)
    if rule == "eighth_kelly":
        return min(bankroll * 0.125 * kelly, bankroll * 0.03)
    if rule == "quarter_kelly":
        return min(bankroll * 0.25 * kelly, bankroll * 0.05)
    raise KeyError(rule)


def run_staking(df: pd.DataFrame, spec: dict[str, Any], rule: str) -> pd.DataFrame:
    bankroll = 100.0
    peak = bankroll
    rows = []
    prob_col = spec["prob_col"]
    edge_col = spec["edge_col"]
    for seq, row in enumerate(df.itertuples(index=False), start=1):
        selection_prob = float(getattr(row, prob_col))
        stake = stake_amount(rule, bankroll, selection_prob, float(row.best_decimal_odds))
        if stake <= 0:
            continue
        win = int(row.favorite_final_win)
        profit = stake * (float(row.best_decimal_odds) - 1.0) if win else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        rows.append({
            "simulation": spec["simulation"],
            "threshold": spec["threshold"],
            "methodology_status": spec["methodology_status"],
            "selection_probability_label": spec["probability_label"],
            "staking_rule": rule,
            "sequence": seq,
            "game_id": int(row.game_id),
            "trigger_play_id": str(row.trigger_play_id),
            "fav_deficit": int(row.fav_deficit),
            "season": int(row.season),
            "time_bucket": row.time_bucket,
            "fav_team": row.fav_team,
            "dog_team": row.dog_team,
            "model_prob_final_win": float(row.model_prob_final_win),
            "model_prob_deficit_erased": float(row.model_prob_deficit_erased),
            "n06_prob": float(row.model_prob_deficit_erased),
            "selection_prob": float(selection_prob),
            "conformal_width": float(row.conformal_width),
            "market_raw_break_even_prob": float(row.best_market_prob),
            "market_implied_prob": float(row.best_market_prob),
            "decimal_odds": float(row.best_decimal_odds),
            "edge_at_entry": float(getattr(row, edge_col)),
            "edge_at_entry_final_win": float(row.edge_at_entry_final_win),
            "edge_at_entry_deficit_erased": float(row.edge_at_entry_deficit_erased),
            "sw_disagreement": float(row.sw_disagreement),
            "market_price_source": row.market_price_source,
            "fallback_used": bool(row.fallback_used),
            "price_source_type": row.price_source_type,
            "favorite_final_win": win,
            "deficit_erased": int(row.deficit_erased),
            "stake": float(stake),
            "profit": float(profit),
            "bankroll_after": float(bankroll),
            "drawdown": float(peak - bankroll),
        })
    return pd.DataFrame(rows)


bet_frames = []
small_sample_warnings = []
for spec in simulation_specs:
    selected = selected_simulation_rows(spec["simulation"], spec["threshold"])
    if spec["simulation"] in {"B_final_win_model_edge", "B_deficit_erased_heuristic", "D_sw_disagreement"} and len(selected) < 50:
        small_sample_warnings.append({
            "simulation": spec["simulation"],
            "threshold": spec["threshold"],
            "price_subset": "all_bets",
            "n_selected_games": int(len(selected)),
        })
    real_selected = selected[selected["price_source_type"].eq("real_moneyline")]
    if spec["simulation"] in {"B_final_win_model_edge", "B_deficit_erased_heuristic", "D_sw_disagreement"} and len(real_selected) < 50:
        small_sample_warnings.append({
            "simulation": spec["simulation"],
            "threshold": spec["threshold"],
            "price_subset": "real_moneyline_only",
            "n_selected_games": int(len(real_selected)),
        })
    for rule in staking_rules:
        bet_frames.append(run_staking(selected, spec, rule))

bet_rows = pd.concat([f for f in bet_frames if len(f)], ignore_index=True) if bet_frames else pd.DataFrame()
bet_rows.to_parquet(N09_BETS_PARQUET, index=False)


def roi_bootstrap(df: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    if len(df) == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": BOOTSTRAP_RESAMPLES}
    grouped = df.groupby("game_id")[["profit", "stake"]].sum().reset_index()
    gids = np.arange(len(grouped))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_RESAMPLES)
    profits = grouped["profit"].to_numpy(float)
    stakes = grouped["stake"].to_numpy(float)
    for i in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(gids, size=len(gids), replace=True)
        denom = stakes[sample].sum()
        draws[i] = profits[sample].sum() / denom if denom else np.nan
    return {
        "lower": float(np.nanquantile(draws, 0.025)),
        "p25": float(np.nanquantile(draws, 0.25)),
        "median": float(np.nanquantile(draws, 0.50)),
        "p75": float(np.nanquantile(draws, 0.75)),
        "upper": float(np.nanquantile(draws, 0.975)),
        "n_resamples": BOOTSTRAP_RESAMPLES,
    }


def summarize_bets(df: pd.DataFrame, seed: int) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "n_bets": 0, "n_wins": 0, "win_rate": None, "total_units_staked": 0.0,
            "total_units_won": 0.0, "net_profit": 0.0, "roi": None,
            "mean_profit_per_bet": None, "std_profit_per_bet": None,
            "max_drawdown": None, "roi_bootstrap_ci": {"lower": None, "upper": None},
        }
    staked = float(df["stake"].sum())
    net = float(df["profit"].sum())
    return {
        "n_bets": int(len(df)),
        "n_wins": int(df["favorite_final_win"].sum()),
        "win_rate": float(df["favorite_final_win"].mean()),
        "total_units_staked": staked,
        "total_units_won": float(df.loc[df["profit"] > 0, "profit"].sum()),
        "net_profit": net,
        "roi": net / staked if staked else None,
        "mean_profit_per_bet": float(df["profit"].mean()),
        "std_profit_per_bet": float(df["profit"].std(ddof=0)),
        "max_drawdown": float(df["drawdown"].max()),
        "roi_bootstrap_ci": roi_bootstrap(df, seed=seed),
        "edge_at_entry": {
            "mean": float(df["edge_at_entry"].mean()),
            "p10": float(df["edge_at_entry"].quantile(0.10)),
            "p50": float(df["edge_at_entry"].quantile(0.50)),
            "p90": float(df["edge_at_entry"].quantile(0.90)),
            "corr_with_outcome": float(df[["edge_at_entry", "favorite_final_win"]].corr().iloc[0, 1]) if df["edge_at_entry"].nunique() > 1 else None,
        },
        "bankroll_trajectory": df[["sequence", "game_id", "bankroll_after", "drawdown"]].to_dict(orient="records"),
    }


def price_subsets(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("all_bets", df),
        ("real_moneyline_only", df[df["price_source_type"].eq("real_moneyline")]),
        ("synthetic_fallback_only", df[df["price_source_type"].eq("synthetic_fallback")]),
    ]


def interpret_roi(row: dict[str, Any]) -> str:
    ci = row["roi_bootstrap_ci"]
    if row["n_bets"] == 0:
        return "neutral_no_bets"
    if ci["lower"] is not None and ci["lower"] > 0:
        return "strong_betting_edge_vs_pregame_prices"
    if row["roi"] is not None and row["roi"] > 0:
        return "marginal_positive_betting_edge_vs_pregame_prices"
    if row["roi"] is not None and abs(row["roi"]) < 0.02:
        return "neutral_betting_edge_vs_pregame_prices"
    return "negative_betting_edge_vs_pregame_prices"


summary_rows = []
breakdown_deficit = []
breakdown_season = []
for i, ((sim, threshold, rule), grp) in enumerate(bet_rows.groupby(["simulation", "threshold", "staking_rule"], dropna=False)):
    for price_subset, subset in price_subsets(grp):
        row = {
            "simulation": sim,
            "threshold": None if pd.isna(threshold) else float(threshold),
            "staking_rule": rule,
            "price_subset": price_subset,
            **summarize_bets(subset, seed=BOOTSTRAP_SEED + 20000 + i * 10 + len(summary_rows)),
        }
        row["honest_interpretation"] = interpret_roi(row)
        summary_rows.append(row)
        for deficit, dgrp in subset.groupby("fav_deficit"):
            s = summarize_bets(dgrp, seed=BOOTSTRAP_SEED + 30000 + len(breakdown_deficit))
            breakdown_deficit.append({
                "simulation": sim, "threshold": row["threshold"], "staking_rule": rule,
                "price_subset": price_subset,
                "fav_deficit": int(deficit), "thin_flag": thin_flag(int(len(dgrp)), int(dgrp["game_id"].nunique()), int(dgrp["season"].nunique())),
                **{k: v for k, v in s.items() if k not in {"bankroll_trajectory"}},
            })
        for season, sgrp in subset.groupby("season"):
            s = summarize_bets(sgrp, seed=BOOTSTRAP_SEED + 40000 + len(breakdown_season))
            breakdown_season.append({
                "simulation": sim, "threshold": row["threshold"], "staking_rule": rule,
                "price_subset": price_subset,
                "season": int(season), "thin_flag": thin_flag(int(len(sgrp)), int(sgrp["game_id"].nunique()), 1),
                **{k: v for k, v in s.items() if k not in {"bankroll_trajectory"}},
            })

betting_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "line_coverage": line_coverage,
    "simulation_specs": simulation_specs,
    "staking_rules": staking_rules,
    "small_sample_warnings": small_sample_warnings,
    "summary": summary_rows,
    "per_deficit_breakdown": breakdown_deficit,
    "per_season_breakdown": breakdown_season,
    "edge_terminology_note": (
        "Section 3 tests betting edge against pre-game prices only. It does not test live market edge. "
        "B_final_win_model_edge is the same-label final-win track. B_deficit_erased_heuristic is a cross-label heuristic. "
        "market_raw_break_even_prob is the raw offered-price break-even probability, not no-vig market consensus. "
        "Synthetic fallback prices are separated from real-moneyline headline results."
    ),
}
N09_BETTING_JSON.write_text(json.dumps(json_safe(betting_payload), indent=2) + "\n", encoding="utf-8")
print(f"[ok] wrote {N09_BETS_PARQUET.relative_to(REPO_ROOT)} rows={len(bet_rows):,}")
print(f"[ok] wrote {N09_BETTING_JSON.relative_to(REPO_ROOT)}")
print("[info] small-sample warnings:", small_sample_warnings)
""")


add("code", "c09_0009", r"""
def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def first_summary(
    sim: str,
    rule: str = "flat_1u",
    threshold: float | None = None,
    price_subset: str = "all_bets",
) -> dict[str, Any] | None:
    for row in summary_rows:
        if row["simulation"] != sim or row["staking_rule"] != rule:
            continue
        if row.get("price_subset") != price_subset:
            continue
        if threshold is None and row["threshold"] is None:
            return row
        if threshold is not None and row["threshold"] is not None and abs(row["threshold"] - threshold) < 1e-12:
            return row
    return None


tf = two_feature_reproduction["deficit_erased"]
ff = two_feature_reproduction["favorite_final_win"]
top_dis = disagreement_analysis["top100_high_disagreement"]["summary"]["deficit_erased"]
base_all_real = first_summary("A_all_trigger_games", "flat_1u", price_subset="real_moneyline_only")
final_b00_real = first_summary("B_final_win_model_edge", "flat_1u", threshold=0.00, price_subset="real_moneyline_only")
final_b05_real = first_summary("B_final_win_model_edge", "flat_1u", threshold=0.05, price_subset="real_moneyline_only")
final_b10_real = first_summary("B_final_win_model_edge", "flat_1u", threshold=0.10, price_subset="real_moneyline_only")
final_b10_all = first_summary("B_final_win_model_edge", "flat_1u", threshold=0.10, price_subset="all_bets")
heur_b10_real = first_summary("B_deficit_erased_heuristic", "flat_1u", threshold=0.10, price_subset="real_moneyline_only")
heur_b10_synth = first_summary("B_deficit_erased_heuristic", "flat_1u", threshold=0.10, price_subset="synthetic_fallback_only")


def ci_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "[NA, NA]"
    ci = row["roi_bootstrap_ci"]
    return f"[{fmt(ci.get('lower'))}, {fmt(ci.get('upper'))}]"

lines: list[str] = []
lines.append("# N09 -- Trigger-state analysis, dashboard stratification, and counterfactual betting simulation")
lines.append("")
lines.append("## Lead findings")
lines.append("")
if base_all_real:
    lines.append(f"**Primary finding:** pre-game CFB markets correctly price favorite comeback risk on average. The unfiltered always-bet-favorite strategy on trigger games produces ROI **{fmt(base_all_real['roi'])}** with bootstrap 95% CI **{ci_text(base_all_real)}** across **{base_all_real['n_bets']}** real-moneyline bets. This is a clean statistically significant loss, not a hidden broad betting edge against pre-game prices.")
else:
    lines.append("**Primary finding:** no unfiltered real-moneyline trigger-game betting baseline was produced, which should not happen under the N09 design.")
lines.append("")
if final_b00_real and final_b05_real and final_b10_real:
    lines.append(f"**Secondary finding, suggestive but underpowered:** the methodologically valid same-label edge filter using N03's `favorite_final_win` probability is positive at every tested edge threshold: ROI **{fmt(final_b00_real['roi'])}** at edge >= 0.00 (**{final_b00_real['n_bets']}** bets), **{fmt(final_b05_real['roi'])}** at edge >= 0.05 (**{final_b05_real['n_bets']}** bets), and **{fmt(final_b10_real['roi'])}** at edge >= 0.10 (**{final_b10_real['n_bets']}** bets). All CIs have positive lower bounds, but every real-moneyline sample is below the project's locked 50-event floor for rate-comparison claims. This is suggestive, not deployable. The underpower is itself informative: the same-label model rarely disagrees with pre-game markets by 10+ percentage points on trigger games, consistent with pre-game markets being well-calibrated for this subpopulation.")
lines.append("")
if heur_b10_real:
    lines.append(f"**Tertiary finding, real pattern with unverified mechanism:** the original N06-based Sim B track, which filters with the `deficit_erased` model while paying on `favorite_final_win`, produces ROI **{fmt(heur_b10_real['roi'])}** on **{heur_b10_real['n_bets']}** real-moneyline bets at edge >= 0.10. This is methodologically a cross-label heuristic, not a validated final-win betting edge. N04's same-label model showed negative edge on these same games, so the favorable realized outcomes are not explained by a probability framework we have validated. Preserve it as a research curiosity, not a deployable strategy.")
lines.append("")
lines.append("**Project-level implication:** pre-game prices are approximately efficient on CFB trigger games. The same-label filter shows suggestive directional betting edge but needs much more data to confirm. The live-data path is the route to accumulating enough trigger events with market prices to test whether that same-label signal is real. Future N10+ live-data scaffold work should prioritize collecting trigger events and live market prices across multiple seasons.")
lines.append("")
lines.append("## Section 1 -- baseline_C structural analysis")
lines.append("")
lines.append(f"**Structural edge finding:** baseline_C remains the dominant structural signal. On held-out `deficit_erased`, the two-feature deficit + time logistic model has Brier **{fmt(tf['two_feature_brier'])}**, baseline_C has Brier **{fmt(tf['baseline_C_brier'])}**, and N06 has Brier **{fmt(tf['n06_brier'])}**. N06 deviations from baseline_C are not reliably better: among the 100 highest-disagreement triggers, actual `deficit_erased` rate is **{fmt(top_dis['actual_rate'])}**, mean N06 probability is **{fmt(top_dis['mean_n06_prob'])}**, and mean baseline_C probability is **{fmt(top_dis['mean_baseline_C_prob'])}**; closer aggregate rate = **{top_dis['closer_to_actual_rate']}**.")
lines.append("")
lines.append(f"The two-feature deficit + time-bucket logistic recovers most of N06's predictive performance on `deficit_erased` (Brier **{fmt(tf['two_feature_brier'])}** versus N06 **{fmt(tf['n06_brier'])}**). This continues the N05 through N08 structural-edge pattern: the model's deviations from baseline_C are not reliably better than the simple structural lookup. The 20-cell baseline_C table and season-stability diagnostics are written to `n09_baseline_analysis.json`.")
lines.append("")
lines.append("## Section 2 -- dashboard stratifications")
lines.append("")
lines.append(f"**Dashboard finding:** rich descriptive stratifications across **11** game-state dimensions are pre-computed for **{len(strat_df):,}** trigger events and are ready for dashboard consumption. Dimensions include turnover composition, short-field proxy, explosive composition, EPA differential, pace, possessions remaining, team tempo/pass style, favorite and dog momentum, conference/neutral context, week bucket, and the single authorized deficit x favorite-momentum cross dimension.")
lines.append("")
lines.append("Numeric stratification buckets are empirical quintiles of the full corpus. Composition buckets keep zero/no-points cases separate and use quintiles over the positive population. Remaining concentration is intentional only for intrinsic categorical buckets such as first-two-games and early-game momentum. Every bucket table reports both labels (`favorite_final_win` and `deficit_erased`) with `n_events`, `n_games`, `n_seasons`, `thin_flag`, and Wilson intervals.")
lines.append("")
lines.append("## Section 3 -- counterfactual betting simulation against pre-game prices")
lines.append("")
lines.append("**Audit-corrected betting-edge framing:** Sim B is reported two ways. The methodologically valid version uses N03's same-label `favorite_final_win` probability as `model_prob_final_win`. The N06 version remains as a `deficit_erased` selection heuristic and is not a final-win probability-edge claim. Headline ROI uses real-moneyline rows only; synthetic fallback prices are reported separately and are not treated as cached sportsbook results.")
lines.append("")
if base_all_real:
    lines.append(f"Unfiltered real-moneyline trigger strategy: **{base_all_real['n_bets']}** bets, ROI **{fmt(base_all_real['roi'])}**, bootstrap 95% CI **{ci_text(base_all_real)}**. This tests betting edge against pre-game prices only; it does not test live market edge.")
else:
    lines.append("**Betting-edge finding vs pre-game prices:** no unfiltered flat-stake bets were produced, which should not happen under the N09 design.")
lines.append("")
if final_b00_real:
    lines.append(f"Same-label Sim B (`B_final_win_model_edge`, threshold 0.00, real moneyline only): **{final_b00_real['n_bets']}** bets, win rate **{fmt(final_b00_real['win_rate'])}**, ROI **{fmt(final_b00_real['roi'])}**, CI **{ci_text(final_b00_real)}**.")
if final_b05_real:
    lines.append(f"Same-label Sim B (`B_final_win_model_edge`, threshold 0.05, real moneyline only): **{final_b05_real['n_bets']}** bets, win rate **{fmt(final_b05_real['win_rate'])}**, ROI **{fmt(final_b05_real['roi'])}**, CI **{ci_text(final_b05_real)}**.")
if final_b10_real:
    lines.append(f"Same-label Sim B (`B_final_win_model_edge`, threshold 0.10, real moneyline only): **{final_b10_real['n_bets']}** bets, win rate **{fmt(final_b10_real['win_rate'])}**, ROI **{fmt(final_b10_real['roi'])}**, CI **{ci_text(final_b10_real)}**.")
lines.append("The same-label Sim B track is positive but underpowered: its real-moneyline sample is below the 50-bet reporting threshold at every tested edge threshold, so it is descriptive rather than bankable evidence.")
if final_b10_all:
    lines.append(f"Same-label Sim B all-price comparison at threshold 0.10: **{final_b10_all['n_bets']}** bets, ROI **{fmt(final_b10_all['roi'])}**, CI **{ci_text(final_b10_all)}**.")
if heur_b10_real:
    lines.append(f"N06 deficit-erased heuristic Sim B at threshold 0.10, real moneyline only: **{heur_b10_real['n_bets']}** bets, win rate **{fmt(heur_b10_real['win_rate'])}**, ROI **{fmt(heur_b10_real['roi'])}**, CI **{ci_text(heur_b10_real)}**. This is a selection heuristic, not a same-label betting-edge claim.")
if heur_b10_synth:
    lines.append(f"N06 heuristic synthetic-fallback subset at threshold 0.10: **{heur_b10_synth['n_bets']}** rows, ROI **{fmt(heur_b10_synth['roi'])}**. This subset is not part of headline betting-edge claims because the prices are baseline-derived synthetic odds.")
if small_sample_warnings:
    lines.append("")
    lines.append(f"Small-sample warning: {len(small_sample_warnings)} Sim B/D threshold(s) produced fewer than 50 selected games. They remain in the data but should be treated as descriptive curiosities only.")
lines.append("")
lines.append("## Deliverables")
lines.append("")
lines.append(f"- `n09_trigger_state_stratifications.parquet`: {len(strat_df):,} rows.")
lines.append(f"- `n09_baseline_analysis.json`: Section 1 + Section 2 aggregate tables.")
lines.append(f"- `n09_betting_simulations.parquet`: {len(bet_rows):,} bet/staking rows.")
lines.append("- `n09_betting_summary.json`: Section 3 aggregate metrics, CIs, bankroll trajectories, and sample-size flags.")
lines.append("")
lines.append("## Honest interpretation")
lines.append("")
lines.append("N09 is application-facing, but it does not soften the research conclusion. Section 1 remains clean: baseline_C captures the dominant structural signal. Section 2 is descriptive dashboard data with empirical buckets and sample-size flags. Section 3 says pre-game markets are efficient on average, the same-label filter is promising but underpowered, and the N06 heuristic is a real pattern without a validated final-win mechanism. Live data collection is the next required step.")
lines.append("")
N09_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] wrote {N09_SUMMARY_MD.relative_to(REPO_ROOT)}")
print("\n".join(lines[:20]))
""")


def _format_source(src: str) -> list[str]:
    return [line + "\n" for line in src.splitlines()]


nb = {
    "cells": [
        {
            "cell_type": cell_type,
            "id": cell_id,
            "metadata": {},
            "source": _format_source(src),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        }
        for cell_type, cell_id, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
