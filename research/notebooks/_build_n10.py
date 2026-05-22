"""
Deterministic builder for research/notebooks/10_fluke_deficit_conditional_analysis.ipynb.

N10 is a descriptive conditional-rate notebook. It directly tests the
mechanistic fluke-deficit hypothesis against pre-game market probabilities.
No new model is trained.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "10_fluke_deficit_conditional_analysis.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m10_0000", """
# Notebook 10 -- Direct conditional analysis of fluke-deficit comebacks

N10 answers the project's core mechanistic question directly:

> When a favorite falls behind because of fluky scoring, is genuinely the
> better pre-game team, and still has meaningful time remaining, do they win
> at rates above pre-game market expectation?

This notebook does not train a model, optimize calibration, or compare to
baseline_C. It reports conditional rates, confidence intervals, pre-game
no-vig probability comparisons, and held-out pre-game-price ROI diagnostics.

Terminology discipline:

- **Pre-game probability comparison** uses no-vig implied probabilities.
- **Betting simulation** uses raw offered odds because bettors pay the vig.
- **Live market edge is not tested**; N10 only identifies candidate live-watch
  football states for future data collection.
- Both labels are reported everywhere: `favorite_final_win` and
  `deficit_erased`.
""")


add("code", "c10_0001", r"""
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

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
CACHE_DIR = RESEARCH_DIR / "data" / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
N07_FEATURES = RESULTS_DIR / "n07_descriptive_features.parquet"
N09_STRAT = RESULTS_DIR / "n09_trigger_state_stratifications.parquet"
N04_SPEC = RESULTS_DIR / "n04_spec.json"

N10_RATES_PARQUET = RESULTS_DIR / "n10_conditional_rates.parquet"
N10_ANALYSIS_JSON = RESULTS_DIR / "n10_conditional_analysis.json"
N10_SUMMARY_MD = RESULTS_DIR / "n10_summary_report.md"

for path in [TRIGGER_EVENTS_CSV, N07_FEATURES, N09_STRAT, N04_SPEC]:
    assert path.exists(), f"Missing required N10 input: {path}"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
EPS = 1e-12
LABELS = ["favorite_final_win", "deficit_erased"]
BET_YEARS = {2022, 2023, 2024}
SPORTSBOOKS = ["Bovada", "DraftKings", "ESPN Bet", "Caesars", "William Hill (New Jersey)"]
SPORTSBOOK_SET = set(SPORTSBOOKS)
ALGORITHMIC_PROVIDERS = {"teamrankings", "numberfire"}

print(f"[ok] N10 setup at {NOTEBOOK_DIR}")
""")


add("code", "c10_0002", r"""
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
    if not isinstance(obj, (str, bytes)) and pd.isna(obj):
        return None
    return obj


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


def bootstrap_cluster_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(df) == 0 or df[value_col].notna().sum() == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
    work = df[["game_id", value_col]].dropna()
    grouped = {gid: g[value_col].to_numpy(dtype=float) for gid, g in work.groupby("game_id", sort=False)}
    gids = np.array(list(grouped))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(gids, size=len(gids), replace=True)
        vals = np.concatenate([grouped[gid] for gid in sampled])
        draws[i] = float(np.mean(vals))
    return {
        "lower": float(np.nanquantile(draws, 0.025)),
        "p25": float(np.nanquantile(draws, 0.25)),
        "median": float(np.nanquantile(draws, 0.50)),
        "p75": float(np.nanquantile(draws, 0.75)),
        "upper": float(np.nanquantile(draws, 0.975)),
        "n_resamples": n_resamples,
    }


def bootstrap_cluster_roi(
    df: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(df) == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
    work = df[["game_id", "profit", "stake"]].dropna()
    if len(work) == 0 or work["stake"].sum() <= 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
    grouped = work.groupby("game_id", sort=False)[["profit", "stake"]].sum().reset_index()
    profits = grouped["profit"].to_numpy(dtype=float)
    stakes = grouped["stake"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(grouped))
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(idx, size=len(idx), replace=True)
        denom = stakes[sample].sum()
        draws[i] = profits[sample].sum() / denom if denom > 0 else np.nan
    return {
        "lower": float(np.nanquantile(draws, 0.025)),
        "p25": float(np.nanquantile(draws, 0.25)),
        "median": float(np.nanquantile(draws, 0.50)),
        "p75": float(np.nanquantile(draws, 0.75)),
        "upper": float(np.nanquantile(draws, 0.975)),
        "n_resamples": n_resamples,
    }


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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


print("[ok] statistical and odds helpers defined")
""")


add("code", "c10_0003", r"""
def load_json_cache(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CACHE_DIR.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
    return rows


def load_lines_by_id() -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for rec in load_json_cache("cfbd__lines__*.json"):
        gid = int(rec["id"])
        if gid not in by_id or len(rec.get("lines") or []) > len(by_id[gid].get("lines") or []):
            by_id[gid] = rec
    return by_id


def load_drives_by_id(trigger_game_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in load_json_cache("cfbd__drives__*.json"):
        gid = int(rec.get("gameId") or 0)
        if gid in trigger_game_ids:
            by_id[gid].append(rec)
    return dict(by_id)


def load_plays_by_id(trigger_game_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in load_json_cache("cfbd__plays__*.json"):
        gid = int(rec.get("gameId") or 0)
        if gid in trigger_game_ids:
            by_id[gid].append(rec)
    out: dict[int, list[dict[str, Any]]] = {}
    for gid, plays in by_id.items():
        dedup = {str(p.get("id")): p for p in plays}
        out[gid] = sorted(dedup.values(), key=chrono_key)
    return out


def chrono_key(p: dict[str, Any]) -> tuple[int, int, int, int]:
    period = int(p.get("period") or 0)
    clock = p.get("clock") or {}
    m = clock.get("minutes")
    s = clock.get("seconds")
    elapsed = 900 - 60 * int(m) - int(s) if m is not None and s is not None else 0
    return (period, elapsed, int(p.get("driveNumber") or 0), int(p.get("playNumber") or 0))


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


def selected_home_spread(rec: dict[str, Any]) -> dict[str, Any] | None:
    lines = rec.get("lines") or []
    consensus = [ln for ln in lines if ln.get("provider") == "consensus" and has_field(ln, "spread")]
    if consensus:
        return {"home_spread": float(consensus[0]["spread"]), "spread_provider_used": "consensus"}
    vals = []
    providers = []
    for ln in lines:
        if ln.get("provider") in SPORTSBOOK_SET and has_field(ln, "spread"):
            vals.append(float(ln["spread"]))
            providers.append(str(ln.get("provider")))
    if not vals:
        return None
    return {
        "home_spread": float(np.mean(vals)),
        "spread_provider_used": f"single_provider_{providers[0]}" if len(vals) == 1 else "multi_sportsbook_avg",
    }


def spread_favorite_side(home_spread: float | None) -> str | None:
    if home_spread is None or pd.isna(home_spread) or abs(float(home_spread)) < EPS:
        return None
    return "home" if float(home_spread) < 0 else "away"


def favorite_spread_from_home_spread(home_spread: float | None) -> float | None:
    if home_spread is None or pd.isna(home_spread):
        return None
    return -abs(float(home_spread))


n04_spec = json.loads(N04_SPEC.read_text(encoding="utf-8"))
spread_model_spec = n04_spec["spread_conversion_model"]
SPREAD_COEF = float(spread_model_spec["coefficient"])
SPREAD_INTERCEPT = float(spread_model_spec["intercept"])


def spread_converted_favorite_prob(favorite_spread: float) -> float:
    return sigmoid(SPREAD_INTERCEPT + SPREAD_COEF * float(favorite_spread))


lines_by_id = load_lines_by_id()
print(f"[ok] loaded cached line records for {len(lines_by_id):,} games")
""")


add("code", "c10_0004", r"""
def market_prices_for_favorite(game_id: int, fav_team: str) -> dict[str, Any]:
    rec = lines_by_id.get(int(game_id))
    if rec is None:
        return {
            "pregame_raw_implied_prob": np.nan,
            "pregame_no_vig_implied_prob": np.nan,
            "decimal_odds_best_available": np.nan,
            "is_synthetic_fallback_price": True,
            "market_price_source": "no_line_record",
            "vig_status": "missing",
        }

    target_side = team_side_for_game(rec, fav_team)
    spread = selected_home_spread(rec)
    home_spread = None if spread is None else float(spread["home_spread"])
    spread_side = spread_favorite_side(home_spread)
    favorite_spread = favorite_spread_from_home_spread(home_spread)
    target_spread_prob = np.nan
    if favorite_spread is not None and spread_side is not None:
        fav_prob = spread_converted_favorite_prob(favorite_spread)
        target_spread_prob = fav_prob if target_side == spread_side else 1.0 - fav_prob

    consensus_rows: list[dict[str, Any]] = []
    sportsbook_rows: list[dict[str, Any]] = []
    conflict_providers: list[str] = []

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
            conflict_providers.append(str(provider))
            continue
        odds = ln.get("homeMoneyline") if target_side == "home" else ln.get("awayMoneyline")
        raw_prob = american_raw_prob(odds)
        dec = american_decimal_odds(odds)
        no_vig, vig_status = probability_for_side_from_line(ln, target_side)
        if raw_prob is None or dec is None or no_vig is None:
            continue
        row = {
            "provider": str(provider),
            "raw_prob": float(raw_prob),
            "no_vig_prob": float(no_vig),
            "decimal_odds": float(dec),
            "vig_status": vig_status,
        }
        if provider == "consensus":
            consensus_rows.append(row)
        elif provider in SPORTSBOOK_SET:
            sportsbook_rows.append(row)

    selected_prob_rows = consensus_rows if consensus_rows else sportsbook_rows
    if selected_prob_rows:
        prob_source = "consensus" if consensus_rows else (
            f"single_provider_{selected_prob_rows[0]['provider']}" if len(selected_prob_rows) == 1 else "multi_sportsbook_avg"
        )
        no_vig_prob = float(np.mean([r["no_vig_prob"] for r in selected_prob_rows]))
        raw_prob = float(np.mean([r["raw_prob"] for r in selected_prob_rows]))
        vig_statuses = sorted(set(r["vig_status"] for r in selected_prob_rows))
        if sportsbook_rows:
            best = max(sportsbook_rows, key=lambda r: r["decimal_odds"])
            best_dec = float(best["decimal_odds"])
            raw_break_even = float(1.0 / best_dec)
            raw_source = f"best_sportsbook_{best['provider']}"
        else:
            best = selected_prob_rows[0]
            best_dec = float(best["decimal_odds"])
            raw_break_even = float(1.0 / best_dec)
            raw_source = "consensus_only"
        return {
            "pregame_raw_implied_prob": raw_break_even,
            "pregame_no_vig_implied_prob": no_vig_prob,
            "decimal_odds_best_available": best_dec,
            "is_synthetic_fallback_price": False,
            "market_price_source": raw_source,
            "probability_provider_used": prob_source,
            "vig_status": vig_statuses[0] if len(vig_statuses) == 1 else "mixed",
            "moneyline_direction_conflict_providers": sorted(set(conflict_providers)),
        }

    if not pd.isna(target_spread_prob):
        dec = float(1.0 / np.clip(target_spread_prob, EPS, 1 - EPS))
        return {
            "pregame_raw_implied_prob": float(target_spread_prob),
            "pregame_no_vig_implied_prob": float(target_spread_prob),
            "decimal_odds_best_available": dec,
            "is_synthetic_fallback_price": True,
            "market_price_source": "spread_conversion_no_direction_consistent_moneyline",
            "probability_provider_used": "spread_conversion",
            "vig_status": "spread_conversion",
            "moneyline_direction_conflict_providers": sorted(set(conflict_providers)),
        }

    return {
        "pregame_raw_implied_prob": np.nan,
        "pregame_no_vig_implied_prob": np.nan,
        "decimal_odds_best_available": np.nan,
        "is_synthetic_fallback_price": True,
        "market_price_source": "no_usable_moneyline_or_spread",
        "probability_provider_used": "missing",
        "vig_status": "missing",
        "moneyline_direction_conflict_providers": sorted(set(conflict_providers)),
    }


print("[ok] market price helper defined")
""")


add("code", "c10_0005", r"""
n09 = pd.read_parquet(N09_STRAT)
n07 = pd.read_parquet(N07_FEATURES)
triggers = pd.read_csv(TRIGGER_EVENTS_CSV)

key_cols = ["game_id", "fav_deficit", "trigger_sequence"]
n10 = n09.merge(
    n07[[
        "game_id", "fav_deficit", "trigger_sequence", "quarter",
        "dog_points_from_turnovers_pct", "dog_points_from_returns_pct", "dog_points_from_explosives_pct",
        "dog_points_from_explosives_pct_is_null", "success_rate_gap",
    ]],
    on=key_cols,
    how="left",
    validate="one_to_one",
)
n10 = n10.merge(
    triggers[[
        "game_id", "fav_deficit", "trigger_sequence", "pregame_spread",
        "drive_number_in_game", "play_number", "clock_seconds_in_period_total",
    ]],
    on=key_cols,
    how="left",
    validate="one_to_one",
)
assert len(n10) == 11412, f"Expected 11,412 N10 rows, got {len(n10):,}"
assert not n10.columns.duplicated().any(), "Duplicate columns after N10 base merge"
suffix_cols = [c for c in n10.columns if c.endswith("_x") or c.endswith("_y")]
assert not suffix_cols, f"Unexpected merge suffix columns after N10 base merge: {suffix_cols}"

n10["period"] = n10["quarter"].astype(int)
n10["time_bucket"] = "Q" + n10["period"].astype(str)

component_cols = [
    "dog_points_from_turnovers_pct",
    "dog_points_from_returns_pct",
    "dog_points_from_explosives_pct",
]
n10["fluke_component_missing"] = n10[component_cols].isna().any(axis=1)
n10["fluke_composite"] = n10[component_cols].sum(axis=1, skipna=False)
edge_explosive_mask = n10["dog_points_from_explosives_pct_is_null"].eq(1) & n10["dog_score_at_trigger"].gt(0)
edge_missing_count = int((edge_explosive_mask & n10["dog_points_from_explosives_pct"].isna()).sum())
if edge_missing_count:
    # N07 set known impossible >1 explosive percentages to NaN for modeling. For N10
    # classification, those rows are by definition fluky because the explosive bucket
    # exceeded visible dog score.
    n10.loc[edge_explosive_mask & n10["dog_points_from_explosives_pct"].isna(), "fluke_composite"] = 1.0
    n10.loc[edge_explosive_mask, "fluke_component_missing"] = False

conditions = [
    n10["dog_score_at_trigger"].eq(0),
    n10["fluke_component_missing"].fillna(False),
    n10["fluke_composite"].ge(0.60),
    n10["fluke_composite"].ge(0.30) & n10["fluke_composite"].lt(0.60),
    n10["fluke_composite"].lt(0.30),
]
choices = ["no_dog_points", "attribution_unclear", "fluky_lead", "mixed_lead", "sustained_lead"]
n10["fluke_bucket"] = np.select(conditions, choices, default="attribution_unclear")
attribution_unclear_count = int(n10["fluke_bucket"].eq("attribution_unclear").sum())

spread = pd.to_numeric(n10["pregame_spread"], errors="coerce")
n10["spread_bucket"] = np.select(
    [
        spread.isna(),
        spread.le(-14),
        spread.le(-7),
        spread.le(-3),
        spread.le(-0.5),
        spread.gt(-0.5),
    ],
    ["no_spread", "huge_favorite", "big_favorite", "moderate_favorite", "small_favorite", "pick_or_dog"],
    default="no_spread",
)

market_rows = []
for row in n10[["game_id", "fav_team"]].drop_duplicates().itertuples(index=False):
    market_rows.append({"game_id": int(row.game_id), **market_prices_for_favorite(int(row.game_id), str(row.fav_team))})
market_df = pd.DataFrame(market_rows)
n10 = n10.merge(market_df, on="game_id", how="left", validate="many_to_one")

required_market = ["pregame_raw_implied_prob", "pregame_no_vig_implied_prob", "decimal_odds_best_available"]
missing_market_rows = int(n10[required_market].isna().any(axis=1).sum())
assert missing_market_rows == 0, f"Missing market price fields on {missing_market_rows:,} N10 rows"

print(f"[ok] N10 base rows={len(n10):,}; edge explosive rows classified as fluky={edge_missing_count}")
print(f"[info] attribution_unclear rows={attribution_unclear_count:,} (Option C; descriptive only, excluded from headline)")
print("[info] fluke bucket counts:", n10["fluke_bucket"].value_counts().to_dict())
print("[info] spread bucket counts:", n10["spread_bucket"].value_counts().to_dict())
print("[info] market source counts:", n10["market_price_source"].value_counts().to_dict())
""")


add("code", "c10_0006", r"""
EXCLUDED_DN_PLAY_TYPES = {
    "Penalty", "Timeout", "End Period", "End of Half", "End of Game",
    "Two Point Conversion No Good", "Two Point Conversion Failed",
    "Two Point Pass", "Two Point Rush", "placeholder", "Uncategorized",
}


def _completed_drives_before(drives_for_game: list[dict[str, Any]], trig_drive: int, offense: str) -> list[dict[str, Any]]:
    out = []
    for d in drives_for_game:
        dn = d.get("driveNumber")
        if dn is None or int(dn) >= trig_drive:
            continue
        if d.get("offense") == offense:
            out.append(d)
    return out


def _drive_points_for_offense(drive: dict[str, Any]) -> int:
    try:
        return max(0, int(drive.get("endOffenseScore") or 0) - int(drive.get("startOffenseScore") or 0))
    except (TypeError, ValueError):
        return 0


def _yards_per_point(drives_for_game: list[dict[str, Any]], trig_drive: int, offense: str) -> float | None:
    drives = _completed_drives_before(drives_for_game, trig_drive, offense)
    if not drives:
        return None
    yards_sum = 0
    points_sum = 0
    for d in drives:
        if d.get("yards") is not None:
            yards_sum += int(d["yards"])
        points_sum += _drive_points_for_offense(d)
    if points_sum <= 0:
        return None
    return float(yards_sum) / float(points_sum)


def _effective_distance(p: dict[str, Any]) -> int | None:
    try:
        return max(0, min(int(p["distance"]), int(p["yardsToGoal"])))
    except (KeyError, TypeError, ValueError):
        return None


def _early_success_rate(plays_before: list[dict[str, Any]], offense: str) -> float | None:
    den = 0
    succ = 0
    for p in plays_before:
        if p.get("offense") != offense:
            continue
        if str(p.get("playType") or "") in EXCLUDED_DN_PLAY_TYPES:
            continue
        try:
            down = int(p.get("down"))
            yards = int(p.get("yardsGained"))
        except (TypeError, ValueError):
            continue
        if down not in (1, 2):
            continue
        eff = _effective_distance(p)
        if eff is None:
            continue
        needed = math.ceil((0.50 if down == 1 else 0.70) * eff - 1e-12)
        den += 1
        succ += int(yards >= needed)
    return None if den == 0 else succ / den


def _dog_yards_and_plays_before(plays_before: list[dict[str, Any]], offense: str) -> tuple[float | None, int]:
    yards_sum = 0.0
    play_count = 0
    for p in plays_before:
        if p.get("offense") != offense:
            continue
        if str(p.get("playType") or "") in EXCLUDED_DN_PLAY_TYPES:
            continue
        try:
            yards = float(p.get("yardsGained"))
        except (TypeError, ValueError):
            continue
        yards_sum += yards
        play_count += 1
    if play_count == 0:
        return None, 0
    return yards_sum, play_count


trigger_game_ids = set(int(x) for x in n10["game_id"].unique())
drives_by_game = load_drives_by_id(trigger_game_ids)
plays_by_game = load_plays_by_id(trigger_game_ids)
missing_drives = sorted(trigger_game_ids - set(drives_by_game))
missing_plays = sorted(trigger_game_ids - set(plays_by_game))
assert not missing_drives, f"Missing drives for trigger games: {missing_drives[:10]}"
assert not missing_plays, f"Missing plays for trigger games: {missing_plays[:10]}"

sanity_rows: list[dict[str, Any]] = []
for row in n10.itertuples(index=False):
    gid = int(row.game_id)
    trig_key = (
        int(row.period),
        900 - int(row.clock_seconds_in_period_total),
        int(row.drive_number_in_game),
        int(row.play_number),
    )
    plays_before = [p for p in plays_by_game[gid] if chrono_key(p) < trig_key]
    dog_drives_before = _completed_drives_before(drives_by_game[gid], int(row.drive_number_in_game), str(row.dog_team))
    dog_yards_at_trigger, dog_play_count = _dog_yards_and_plays_before(plays_before, str(row.dog_team))
    dog_score = max(1, int(row.dog_score_at_trigger))
    dog_yards_per_point = None if dog_yards_at_trigger is None else float(dog_yards_at_trigger) / float(dog_score)
    dog_yards_per_play = None if not dog_play_count or dog_yards_at_trigger is None else float(dog_yards_at_trigger) / float(dog_play_count)
    dog_drive_ypp = _yards_per_point(drives_by_game[gid], int(row.drive_number_in_game), str(row.dog_team))
    dog_success = _early_success_rate(plays_before, str(row.dog_team))
    sanity_rows.append({
        "game_id": gid,
        "fav_deficit": int(row.fav_deficit),
        "trigger_sequence": int(row.trigger_sequence),
        "dog_yards_at_trigger_sanity": dog_yards_at_trigger,
        "dog_play_count_sanity": int(dog_play_count),
        "dog_drives_before_trigger_sanity": int(len(dog_drives_before)),
        "dog_yards_per_point_sanity": dog_yards_per_point,
        "dog_yards_per_play_sanity": dog_yards_per_play,
        "dog_drive_yards_per_point_diagnostic": dog_drive_ypp,
        "dog_success_rate_sanity": dog_success,
    })
sanity_df = pd.DataFrame(sanity_rows)
n10 = n10.merge(sanity_df, on=key_cols, how="left", validate="one_to_one")

sanity_summary = []
for bucket, grp in n10.groupby("fluke_bucket", sort=True):
    sanity_summary.append({
        "fluke_bucket": bucket,
        "n_events": int(len(grp)),
        "mean_dog_yards_per_point": float(grp["dog_yards_per_point_sanity"].mean(skipna=True)),
        "median_dog_yards_per_point": float(grp["dog_yards_per_point_sanity"].median(skipna=True)),
        "mean_dog_yards_per_play": float(grp["dog_yards_per_play_sanity"].mean(skipna=True)),
        "median_dog_yards_per_play": float(grp["dog_yards_per_play_sanity"].median(skipna=True)),
        "mean_dog_drives_before_trigger": float(grp["dog_drives_before_trigger_sanity"].mean(skipna=True)),
        "mean_dog_drive_yards_per_point_diagnostic": float(grp["dog_drive_yards_per_point_diagnostic"].mean(skipna=True)),
        "median_dog_drive_yards_per_point_diagnostic": float(grp["dog_drive_yards_per_point_diagnostic"].median(skipna=True)),
        "mean_dog_success_rate": float(grp["dog_success_rate_sanity"].mean(skipna=True)),
        "mean_epa_per_play_gap": float(grp["epa_per_play_gap"].mean(skipna=True)),
        "mean_success_rate_gap": float(grp["success_rate_gap"].mean(skipna=True)),
    })
sanity_summary_df = pd.DataFrame(sanity_summary)
sanity_lookup = sanity_summary_df.set_index("fluke_bucket").to_dict(orient="index")

fluky_ypp = sanity_lookup.get("fluky_lead", {}).get("mean_dog_yards_per_point")
sustained_ypp = sanity_lookup.get("sustained_lead", {}).get("mean_dog_yards_per_point")
clear_fluky_drive_ypp_threshold = sanity_lookup.get("sustained_lead", {}).get(
    "median_dog_drive_yards_per_point_diagnostic"
)
assert clear_fluky_drive_ypp_threshold is not None and not pd.isna(clear_fluky_drive_ypp_threshold), (
    "Cannot compute clear_fluky_lead threshold from sustained_lead median drive yards per point"
)
n10["clear_fluky_lead"] = (
    n10["fluke_bucket"].eq("fluky_lead")
    & n10["dog_drive_yards_per_point_diagnostic"].notna()
    & n10["dog_drive_yards_per_point_diagnostic"].le(float(clear_fluky_drive_ypp_threshold))
)
clear_fluky_count = int(n10["clear_fluky_lead"].sum())
assert clear_fluky_count > 0, "clear_fluky_lead guard produced zero rows"

print("[ok] fluke classification diagnostics computed")
print(
    "[info] broad fluky_lead vs sustained_lead mean dog yards/point: "
    f"{fluky_ypp:.3f} vs {sustained_ypp:.3f}; broad bucket kept descriptive only"
)
print(
    "[info] clear_fluky_lead guard: fluke_bucket=fluky_lead AND "
    f"dog_drive_yards_per_point <= sustained_lead median ({float(clear_fluky_drive_ypp_threshold):.3f}); "
    f"rows={clear_fluky_count:,}"
)
print(sanity_summary_df.to_string(index=False))
""")


add("code", "c10_0007", r"""
def flat_bet_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stake"] = 1.0
    out["profit"] = np.where(
        out["favorite_final_win"].astype(int).eq(1),
        out["decimal_odds_best_available"].astype(float) - 1.0,
        -1.0,
    )
    return out


def roi_summary(df: pd.DataFrame, *, include_synthetic: bool, seed: int) -> dict[str, Any]:
    if include_synthetic:
        work = df[df["season"].isin(BET_YEARS)].copy()
        price_subset = "real_moneyline_plus_synthetic_fallback"
    else:
        work = df[df["season"].isin(BET_YEARS) & (~df["is_synthetic_fallback_price"].fillna(False))].copy()
        price_subset = "real_moneyline_only"
    work = work[work["decimal_odds_best_available"].notna()].copy()
    if len(work) == 0:
        return {
            "price_subset": price_subset,
            "n_bets": 0,
            "n_games": 0,
            "win_rate": None,
            "roi": None,
            "bootstrap_ci": {"lower": None, "upper": None},
        }
    work = flat_bet_rows(work)
    return {
        "price_subset": price_subset,
        "n_bets": int(len(work)),
        "n_games": int(work["game_id"].nunique()),
        "n_seasons": int(work["season"].nunique()),
        "win_rate": float(work["favorite_final_win"].mean()),
        "total_staked": float(work["stake"].sum()),
        "net_profit": float(work["profit"].sum()),
        "roi": float(work["profit"].sum() / work["stake"].sum()),
        "bootstrap_ci": bootstrap_cluster_roi(work, seed=seed),
    }


def group_cell_summary(df: pd.DataFrame, group_cols: list[str], *, include_cells: bool = True, seed_base: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), df)]
    for idx, (key, grp) in enumerate(grouped):
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        base_row = {col: (None if pd.isna(val) else val) for col, val in zip(group_cols, key)} if group_cols else {}
        n_events = int(len(grp))
        n_games = int(grp["game_id"].nunique())
        n_seasons = int(grp["season"].nunique())
        cell_flag = thin_flag(n_events, n_games, n_seasons)
        row: dict[str, Any] = {
            **base_row,
            "n_events": n_events,
            "n_games": n_games,
            "n_seasons": n_seasons,
            "thin_flag": cell_flag,
            "mean_pregame_raw_implied_prob": float(grp["pregame_raw_implied_prob"].mean()),
            "mean_pregame_no_vig_implied_prob": float(grp["pregame_no_vig_implied_prob"].mean()),
            "heldout_real_moneyline_flat_roi": roi_summary(grp, include_synthetic=False, seed=BOOTSTRAP_SEED + seed_base + idx * 2),
            "heldout_all_price_flat_roi": roi_summary(grp, include_synthetic=True, seed=BOOTSTRAP_SEED + seed_base + idx * 2 + 1),
        }
        for label in LABELS:
            successes = int(grp[label].sum())
            diff_col = f"{label}_minus_no_vig"
            tmp = grp.copy()
            tmp[diff_col] = tmp[label].astype(float) - tmp["pregame_no_vig_implied_prob"].astype(float)
            diff_ci = bootstrap_cluster_mean(tmp, diff_col, seed=BOOTSTRAP_SEED + seed_base + 10_000 + idx * 7 + (0 if label == "favorite_final_win" else 1))
            row[label] = {
                "successes": successes,
                "rate": float(successes / n_events) if n_events else None,
                "wilson_ci": wilson_ci(successes, n_events),
                "actual_minus_no_vig": float(tmp[diff_col].mean()),
                "actual_minus_no_vig_bootstrap_ci": diff_ci,
            }
        final_diff = row["favorite_final_win"]["actual_minus_no_vig"]
        final_diff_ci = row["favorite_final_win"]["actual_minus_no_vig_bootstrap_ci"]
        real_roi = row["heldout_real_moneyline_flat_roi"]
        has_unclear_attr = any(
            col in row and row.get(col) == "attribution_unclear"
            for col in ["fluke_bucket"]
        )
        row["candidate_live_watch"] = bool(
            cell_flag == "reliable"
            and not has_unclear_attr
            and final_diff is not None
            and final_diff >= 0.05
            and final_diff_ci.get("lower") is not None
            and final_diff_ci["lower"] > 0
            and real_roi.get("roi") is not None
            and real_roi["roi"] > 0
            and real_roi.get("bootstrap_ci", {}).get("lower") is not None
            and real_roi["bootstrap_ci"]["lower"] > 0
        )
        rows.append(row)
    return rows


def subset_direct_hypothesis(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["clear_fluky_lead"].eq(True)
        & df["spread_bucket"].isin(["huge_favorite", "big_favorite"])
        & df["time_bucket"].isin(["Q1", "Q2"])
    ].copy()


def subset_inverse(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["fluke_bucket"].eq("sustained_lead")
        & df["spread_bucket"].eq("small_favorite")
        & df["time_bucket"].eq("Q4")
    ].copy()


print("[ok] aggregate helpers defined")
""")


add("code", "c10_0008", r"""
tier4 = {
    "fluke_bucket": group_cell_summary(n10, ["fluke_bucket"], seed_base=100),
    "spread_bucket": group_cell_summary(n10, ["spread_bucket"], seed_base=300),
    "time_bucket": group_cell_summary(n10, ["time_bucket"], seed_base=500),
    "deficit": group_cell_summary(n10, ["fav_deficit"], seed_base=700),
}

tier25 = {
    "fluke_x_spread": group_cell_summary(n10, ["fluke_bucket", "spread_bucket"], seed_base=1000),
    "fluke_x_time": group_cell_summary(n10, ["fluke_bucket", "time_bucket"], seed_base=2000),
    "fluke_x_deficit": group_cell_summary(n10, ["fluke_bucket", "fav_deficit"], seed_base=3000),
    "spread_x_time": group_cell_summary(n10, ["spread_bucket", "time_bucket"], seed_base=4000),
    "deficit_x_time": group_cell_summary(n10, ["fav_deficit", "time_bucket"], seed_base=5000),
}

tier1 = group_cell_summary(n10, ["fluke_bucket", "fav_deficit", "time_bucket"], seed_base=6000)
tier2 = group_cell_summary(n10, ["fluke_bucket", "spread_bucket", "time_bucket"], seed_base=8000)
tier3 = group_cell_summary(n10, ["fluke_bucket", "spread_bucket", "fav_deficit", "time_bucket"], seed_base=10000)

direct_df = subset_direct_hypothesis(n10)
inverse_df = subset_inverse(n10)
direct_result = group_cell_summary(direct_df, [], seed_base=12000)[0] if len(direct_df) else {
    "n_events": 0, "n_games": 0, "n_seasons": 0, "thin_flag": "unreliable"
}
direct_result["definition"] = (
    "clear_fluky_lead=True AND spread_bucket in {huge_favorite,big_favorite} "
    "AND time_bucket in {Q1,Q2}; clear_fluky_lead requires fluke_bucket=fluky_lead "
    f"and dog completed-drive yards/point <= sustained_lead median ({float(clear_fluky_drive_ypp_threshold):.3f})"
)
direct_result["subcell_table"] = group_cell_summary(direct_df, ["spread_bucket", "time_bucket"], seed_base=12500) if len(direct_df) else []

inverse_result = group_cell_summary(inverse_df, [], seed_base=13000)[0] if len(inverse_df) else {
    "n_events": 0, "n_games": 0, "n_seasons": 0, "thin_flag": "unreliable"
}
inverse_result["definition"] = "fluke_bucket=sustained_lead AND spread_bucket=small_favorite AND time_bucket=Q4"

candidate_cells = []
for name, rows in [
    ("tier1_fluke_deficit_time", tier1),
    ("tier2_fluke_spread_time", tier2),
    ("tier3_dashboard_only", tier3),
]:
    for row in rows:
        if row.get("candidate_live_watch"):
            candidate_cells.append({"tier": name, **row})

direct_rate = direct_result.get("favorite_final_win", {}).get("rate")
direct_market = direct_result.get("mean_pregame_no_vig_implied_prob")
direct_diff = direct_result.get("favorite_final_win", {}).get("actual_minus_no_vig")
direct_diff_ci = direct_result.get("favorite_final_win", {}).get("actual_minus_no_vig_bootstrap_ci", {})
direct_roi = direct_result.get("heldout_real_moneyline_flat_roi", {})
direct_roi_ci = direct_roi.get("bootstrap_ci", {})

if direct_result.get("thin_flag") != "reliable":
    interpretation_class = "MARGINAL_EVIDENCE_UNDERPOWERED"
elif (
    direct_diff is not None
    and direct_diff_ci.get("lower") is not None
    and direct_diff_ci["lower"] > 0
    and direct_roi.get("roi") is not None
    and direct_roi_ci.get("lower") is not None
    and direct_roi_ci["lower"] > 0
):
    interpretation_class = "STRONG_SUPPORT"
elif direct_diff is not None and direct_diff > 0:
    interpretation_class = "MARGINAL_EVIDENCE"
else:
    interpretation_class = "NULL_OR_NEGATIVE"

inverse_roi = inverse_result.get("heldout_real_moneyline_flat_roi", {})
inverse_ci = inverse_roi.get("bootstrap_ci", {})
inverse_methodology_warning = bool(
    inverse_roi.get("roi") is not None
    and inverse_roi["roi"] > 0
    and inverse_ci.get("lower") is not None
    and inverse_ci["lower"] > 0
)

analysis_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "methodology": {
        "scope": "Full 2015-2024 trigger-event corpus for descriptive analysis; held-out 2022-2024 only for ROI diagnostics.",
        "models_trained": 0,
        "pregame_vs_live_note": "N10 uses pre-game odds only. Candidate live-watch flags are not betting recommendations.",
        "fluke_edge_case_note": f"{int(edge_missing_count)} N07 explosive-percentage edge rows were assigned to fluky_lead because their explosive bucket exceeded visible dog score.",
        "attribution_policy": (
            "Option C hybrid: rows with computable fluke components are classified as fluky/mixed/sustained; "
            "rows with genuinely missing attribution components are assigned to attribution_unclear, reported descriptively, "
            "and excluded from candidate live-watch flags. The direct headline hypothesis test uses a stricter "
            "clear_fluky_lead guard so broad N07-attributed fluky rows are not treated as football-obvious cheap scoring."
        ),
        "attribution_unclear_rows": int(attribution_unclear_count),
        "clear_fluky_lead_policy": {
            "definition": "fluke_bucket == fluky_lead and dog completed-drive yards per point <= sustained_lead median",
            "dog_drive_yards_per_point_threshold": float(clear_fluky_drive_ypp_threshold),
            "clear_fluky_lead_rows": int(clear_fluky_count),
            "reason": (
                "N10 fluke-classification investigation found broad fluky_lead does not separate aggregate yards-per-point "
                "cleanly; the guard preserves N07 component evidence while requiring a direct cheap-yardage signature for "
                "the headline hypothesis cell."
            ),
        },
        "spread_conversion_model": spread_model_spec,
    },
    "sanity_check_diagnostics": sanity_summary_df.to_dict(orient="records"),
    "tier4_single_dimension_splits": tier4,
    "tier25_two_way_matrices": tier25,
    "tier1_fluke_deficit_time": tier1,
    "tier2_fluke_spread_time": tier2,
    "tier3_dashboard_only": tier3,
    "direct_hypothesis_test": direct_result,
    "inverse_hypothesis_sanity_check": inverse_result,
    "inverse_methodology_warning": inverse_methodology_warning,
    "candidate_live_watch_cells": candidate_cells,
    "interpretation_class": interpretation_class,
}

n10_out = n10[[
    "game_id", "trigger_play_id", "fav_deficit", "period", "time_bucket", "trigger_sequence",
    "season", "fav_team", "dog_team", "fluke_composite", "fluke_bucket", "clear_fluky_lead",
    "dog_drive_yards_per_point_diagnostic", "spread_bucket",
    "favorite_final_win", "deficit_erased", "pregame_raw_implied_prob",
    "pregame_no_vig_implied_prob", "decimal_odds_best_available",
    "is_synthetic_fallback_price", "market_price_source", "vig_status",
]].copy()
n10_out.to_parquet(N10_RATES_PARQUET, index=False)
N10_ANALYSIS_JSON.write_text(json.dumps(json_safe(analysis_payload), indent=2) + "\n", encoding="utf-8")

print(f"[ok] wrote {N10_RATES_PARQUET.relative_to(REPO_ROOT)} rows={len(n10_out):,}")
print(f"[ok] wrote {N10_ANALYSIS_JSON.relative_to(REPO_ROOT)}")
print(f"[info] interpretation_class={interpretation_class}")
print(f"[info] direct hypothesis n_events={direct_result.get('n_events')} n_games={direct_result.get('n_games')} thin={direct_result.get('thin_flag')}")
if inverse_methodology_warning:
    print("[warn] inverse hypothesis sanity check has positive real-moneyline ROI with CI lower > 0")
""")


add("code", "c10_0009", r"""
def fmt(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x) * 100:.1f}%" if pct else f"{float(x):+.4f}"


def fmt_rate(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x) * 100:.1f}%"


def ci_text(ci: dict[str, Any], pct: bool = True) -> str:
    lo = ci.get("lower")
    hi = ci.get("upper")
    if lo is None or hi is None:
        return "[NA, NA]"
    return f"[{fmt_rate(lo) if pct else fmt(lo)}, {fmt_rate(hi) if pct else fmt(hi)}]"


def row_label(row: dict[str, Any], cols: list[str]) -> str:
    return " / ".join(str(row.get(c)) for c in cols)


def markdown_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int | None = None) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return out


direct_fw = direct_result.get("favorite_final_win", {})
direct_de = direct_result.get("deficit_erased", {})
direct_real_roi = direct_result.get("heldout_real_moneyline_flat_roi", {})
direct_all_roi = direct_result.get("heldout_all_price_flat_roi", {})
inverse_fw = inverse_result.get("favorite_final_win", {})
inverse_real_roi = inverse_result.get("heldout_real_moneyline_flat_roi", {})

lines: list[str] = []
lines.append("# N10 -- Direct conditional analysis of fluke-deficit comebacks")
lines.append("")

if interpretation_class == "STRONG_SUPPORT":
    lines.append(
        f"**Direct answer: STRONG SUPPORT.** The stricter clear-fluky compound condition is reliable and above pre-game market expectation: favorite final-win rate **{fmt_rate(direct_fw.get('rate'))}** vs mean no-vig market probability **{fmt_rate(direct_result.get('mean_pregame_no_vig_implied_prob'))}**, actual-minus-market **{fmt_rate(direct_fw.get('actual_minus_no_vig'))}** with bootstrap CI **{ci_text(direct_fw.get('actual_minus_no_vig_bootstrap_ci', {}))}**. Held-out real-moneyline flat ROI is **{fmt_rate(direct_real_roi.get('roi'))}** with CI **{ci_text(direct_real_roi.get('bootstrap_ci', {}))}**. This is a candidate live-watch condition, not proof of live market edge."
    )
elif interpretation_class.startswith("MARGINAL"):
    lines.append(
        f"**Direct answer: MARGINAL / UNDERPOWERED.** The stricter clear-fluky compound condition has **{direct_result.get('n_events')}** events across **{direct_result.get('n_games')}** games and **{direct_result.get('n_seasons')}** seasons (`{direct_result.get('thin_flag')}`). Favorite final-win rate is **{fmt_rate(direct_fw.get('rate'))}** vs mean no-vig market probability **{fmt_rate(direct_result.get('mean_pregame_no_vig_implied_prob'))}**; actual-minus-market is **{fmt_rate(direct_fw.get('actual_minus_no_vig'))}** with bootstrap CI **{ci_text(direct_fw.get('actual_minus_no_vig_bootstrap_ci', {}))}**. Held-out real-moneyline flat ROI is **{fmt_rate(direct_real_roi.get('roi'))}** on **{direct_real_roi.get('n_bets')}** bets with CI **{ci_text(direct_real_roi.get('bootstrap_ci', {}))}**. This is worth tracking as a candidate live-watch state only if sample size and live prices can be collected later."
    )
else:
    lines.append(
        f"**Direct answer: NULL OR NEGATIVE.** The stricter clear-fluky compound condition does not show validated underpricing by pre-game markets. Favorite final-win rate is **{fmt_rate(direct_fw.get('rate'))}** vs mean no-vig market probability **{fmt_rate(direct_result.get('mean_pregame_no_vig_implied_prob'))}**; actual-minus-market is **{fmt_rate(direct_fw.get('actual_minus_no_vig'))}** with bootstrap CI **{ci_text(direct_fw.get('actual_minus_no_vig_bootstrap_ci', {}))}**. Edge, if it exists, must come from live in-game prices that N10 cannot test."
    )
lines.append("")
lines.append("N10 uses pre-game odds only. It never tests live market edge. Positive cells are candidate live-watch conditions for future collection, not betting recommendations.")
lines.append("")

lines.append("## Fluke Classification Sanity Checks")
lines.append("")
lines.append("The broad N07-attributed `fluky_lead` bucket did **not** separate aggregate yards-per-point cleanly from `sustained_lead`; this is why N10 uses Option C with a stricter headline guard. The broad bucket remains in descriptive/dashboard tables, while the direct Tier 7 hypothesis test uses `clear_fluky_lead`: `fluke_bucket=fluky_lead` plus dog completed-drive yards per point at or below the `sustained_lead` median. Dog early-down success rate is reported diagnostically rather than used as a gate.")
lines.append(f"`clear_fluky_lead` threshold: dog completed-drive yards per point <= **{float(clear_fluky_drive_ypp_threshold):.3f}**; rows flagged clear-fluky: **{clear_fluky_count:,}**.")
if attribution_unclear_count:
    lines.append(f"Option C attribution handling assigned **{attribution_unclear_count:,}** rows to `attribution_unclear`. These rows remain in descriptive tables but are excluded from the headline fluky-lead hypothesis test and candidate live-watch flags.")
lines.extend(markdown_table(sanity_summary_df.to_dict(orient="records"), [
    "fluke_bucket", "n_events", "mean_dog_yards_per_point", "median_dog_yards_per_point", "mean_dog_drive_yards_per_point_diagnostic", "median_dog_drive_yards_per_point_diagnostic", "mean_dog_success_rate", "mean_epa_per_play_gap", "mean_success_rate_gap"
]))
lines.append("")

lines.append("## Tier 4 Reliable Single-Dimension Context")
lines.append("")
for dim, rows in tier4.items():
    lines.append(f"### {dim}")
    lines.append("")
    table_rows = []
    for r in rows:
        table_rows.append({
            "bucket": row_label(r, [dim]) if dim in r else row_label(r, ["fav_deficit"]),
            "n_events": r["n_events"],
            "n_games": r["n_games"],
            "thin_flag": r["thin_flag"],
            "final_win": fmt_rate(r["favorite_final_win"]["rate"]),
            "deficit_erased": fmt_rate(r["deficit_erased"]["rate"]),
            "no_vig": fmt_rate(r["mean_pregame_no_vig_implied_prob"]),
            "final_minus_market": fmt_rate(r["favorite_final_win"]["actual_minus_no_vig"]),
        })
    lines.extend(markdown_table(table_rows, ["bucket", "n_events", "n_games", "thin_flag", "final_win", "deficit_erased", "no_vig", "final_minus_market"]))
    lines.append("")

lines.append("## Tier 7 Direct Hypothesis Test")
lines.append("")
lines.append(f"Definition: `{direct_result['definition']}`.")
lines.append("")
lines.append(f"- Events/games/seasons: **{direct_result.get('n_events')} / {direct_result.get('n_games')} / {direct_result.get('n_seasons')}** (`{direct_result.get('thin_flag')}`).")
lines.append(f"- `favorite_final_win`: **{fmt_rate(direct_fw.get('rate'))}**, Wilson CI **{ci_text(direct_fw.get('wilson_ci', {}))}**, actual-minus-no-vig **{fmt_rate(direct_fw.get('actual_minus_no_vig'))}**, bootstrap CI **{ci_text(direct_fw.get('actual_minus_no_vig_bootstrap_ci', {}))}**.")
lines.append(f"- `deficit_erased`: **{fmt_rate(direct_de.get('rate'))}**, Wilson CI **{ci_text(direct_de.get('wilson_ci', {}))}**, actual-minus-no-vig **{fmt_rate(direct_de.get('actual_minus_no_vig'))}**, bootstrap CI **{ci_text(direct_de.get('actual_minus_no_vig_bootstrap_ci', {}))}**.")
lines.append(f"- Held-out real-moneyline flat ROI: **{fmt_rate(direct_real_roi.get('roi'))}** on **{direct_real_roi.get('n_bets')}** bets, CI **{ci_text(direct_real_roi.get('bootstrap_ci', {}))}**.")
lines.append(f"- Held-out real + synthetic fallback flat ROI: **{fmt_rate(direct_all_roi.get('roi'))}** on **{direct_all_roi.get('n_bets')}** bets, CI **{ci_text(direct_all_roi.get('bootstrap_ci', {}))}**.")
lines.append("")
if direct_result.get("subcell_table"):
    lines.append("Subcells:")
    sub_rows = []
    for r in direct_result["subcell_table"]:
        sub_rows.append({
            "cell": row_label(r, ["spread_bucket", "time_bucket"]),
            "n_events": r["n_events"],
            "n_games": r["n_games"],
            "thin_flag": r["thin_flag"],
            "final_win": fmt_rate(r["favorite_final_win"]["rate"]),
            "no_vig": fmt_rate(r["mean_pregame_no_vig_implied_prob"]),
            "diff": fmt_rate(r["favorite_final_win"]["actual_minus_no_vig"]),
        })
    lines.extend(markdown_table(sub_rows, ["cell", "n_events", "n_games", "thin_flag", "final_win", "no_vig", "diff"]))
    lines.append("")

lines.append("## Tier 8 Inverse Hypothesis Sanity Check")
lines.append("")
lines.append(f"Definition: `{inverse_result['definition']}`.")
lines.append("")
lines.append(f"- Events/games/seasons: **{inverse_result.get('n_events')} / {inverse_result.get('n_games')} / {inverse_result.get('n_seasons')}** (`{inverse_result.get('thin_flag')}`).")
lines.append(f"- `favorite_final_win`: **{fmt_rate(inverse_fw.get('rate'))}**, actual-minus-no-vig **{fmt_rate(inverse_fw.get('actual_minus_no_vig'))}**.")
lines.append(f"- Held-out real-moneyline flat ROI: **{fmt_rate(inverse_real_roi.get('roi'))}** on **{inverse_real_roi.get('n_bets')}** bets, CI **{ci_text(inverse_real_roi.get('bootstrap_ci', {}))}**.")
if inverse_methodology_warning:
    lines.append("- **Methodology warning:** this inverse cell has positive real-moneyline ROI with bootstrap lower bound above zero; review before making claims.")
lines.append("")

lines.append("## Tier 2.5 Two-Way Findings")
lines.append("")
for name, rows in tier25.items():
    reliable = [r for r in rows if r["thin_flag"] == "reliable"]
    best = sorted(reliable, key=lambda r: r["favorite_final_win"]["actual_minus_no_vig"], reverse=True)[:8]
    lines.append(f"### {name}")
    lines.append("")
    if not best:
        lines.append("No reliable cells.")
    else:
        display_rows = []
        for r in best:
            key_names = [k for k in ["fluke_bucket", "spread_bucket", "fav_deficit", "time_bucket"] if k in r]
            display_rows.append({
                "cell": row_label(r, key_names),
                "n_events": r["n_events"],
                "n_games": r["n_games"],
                "final_win": fmt_rate(r["favorite_final_win"]["rate"]),
                "deficit_erased": fmt_rate(r["deficit_erased"]["rate"]),
                "no_vig": fmt_rate(r["mean_pregame_no_vig_implied_prob"]),
                "diff": fmt_rate(r["favorite_final_win"]["actual_minus_no_vig"]),
            })
        lines.extend(markdown_table(display_rows, ["cell", "n_events", "n_games", "final_win", "deficit_erased", "no_vig", "diff"]))
    lines.append("")

lines.append("## Tier 1 Reliable Three-Way Cells")
lines.append("")
reliable_tier1 = [r for r in tier1 if r["thin_flag"] == "reliable"]
reliable_tier1 = sorted(reliable_tier1, key=lambda r: r["favorite_final_win"]["actual_minus_no_vig"], reverse=True)
if reliable_tier1:
    display_rows = []
    for r in reliable_tier1[:20]:
        display_rows.append({
            "cell": row_label(r, ["fluke_bucket", "fav_deficit", "time_bucket"]),
            "n_events": r["n_events"],
            "n_games": r["n_games"],
            "final_win": fmt_rate(r["favorite_final_win"]["rate"]),
            "deficit_erased": fmt_rate(r["deficit_erased"]["rate"]),
            "no_vig": fmt_rate(r["mean_pregame_no_vig_implied_prob"]),
            "diff": fmt_rate(r["favorite_final_win"]["actual_minus_no_vig"]),
            "candidate_live_watch": r["candidate_live_watch"],
        })
    lines.extend(markdown_table(display_rows, ["cell", "n_events", "n_games", "final_win", "deficit_erased", "no_vig", "diff", "candidate_live_watch"]))
else:
    lines.append("No Tier 1 cells pass the reliable sample-size threshold.")
lines.append("")

lines.append("## Candidate Live-Watch Conditions")
lines.append("")
if candidate_cells:
    lines.append(f"N10 flags **{len(candidate_cells)}** candidate live-watch cells. These are not betting recommendations; they are football states worth collecting live odds on.")
else:
    lines.append("N10 flags **0** candidate live-watch cells under the locked definition.")
lines.append("")

lines.append("## Deliverables")
lines.append("")
lines.append(f"- `n10_conditional_rates.parquet`: {len(n10_out):,} trigger-event rows.")
lines.append("- `n10_conditional_analysis.json`: all tier matrices, direct/inverse tests, sanity diagnostics, and candidate live-watch cells.")
lines.append("- `n10_summary_report.md`: this human-readable report.")
lines.append("")

lines.append("## Honest Interpretation")
lines.append("")
lines.append("N10 is the project's direct conditional answer, but it still uses pre-game prices. If the direct fluky-deficit condition is positive, it means the historical pre-game market underpriced that subset before kickoff; it does not prove that live in-game prices after the favorite falls behind would remain exploitable. If the condition is null or underpowered, that is evidence that pre-game markets already encode much of the favorite-strength and game-context information in these trigger states. The next true market-edge test still requires live odds collection.")
lines.append("")

N10_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"[ok] wrote {N10_SUMMARY_MD.relative_to(REPO_ROOT)}")
print("\n".join(lines[:16]))
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
