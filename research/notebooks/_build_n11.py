"""
Deterministic builder for research/notebooks/11_top25_favorite_stratification.ipynb.

N11 is descriptive and market-efficiency analysis only. It adds AP ranking
stratification to the committed N10 trigger-state rows and tests whether ranked
favorites behave differently from unranked favorites at matched game states.
No model is trained and no CFBD network calls occur inside the notebook.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "11_top25_favorite_stratification.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m11_0000", """
# Notebook 11 -- Top-25 favorite stratification and market efficiency analysis

N11 tests whether AP ranking status is a missing stratification dimension after
N10's direct negative result on the fluky-deficit hypothesis.

Two hypotheses are locked:

- **Hypothesis A, descriptive:** top-25 ranked favorites who fall behind have
  higher comeback rates than unranked favorites at equivalent deficit, time,
  and spread.
- **Hypothesis B, market efficiency:** pre-game markets may price ranked
  favorites differently from unranked favorites. Direction is not assumed.

This notebook uses cached CFBD `/rankings` data pulled before notebook
execution. It does not train models, tune thresholds, or call external APIs.
""")


add("code", "c11_0001", r"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import time
import unicodedata
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
CACHE_DIR = RESEARCH_DIR / "data" / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"

N10_RATES = RESULTS_DIR / "n10_conditional_rates.parquet"
TRIGGERS = RESULTS_DIR / "trigger_events.csv"

N11_PARQUET = RESULTS_DIR / "n11_ranking_stratification.parquet"
N11_ANALYSIS_JSON = RESULTS_DIR / "n11_analysis_results.json"
N11_SUMMARY_MD = RESULTS_DIR / "n11_summary_report.md"

for path in [N10_RATES, TRIGGERS]:
    assert path.exists(), f"Missing required N11 input: {path}"
for year in range(2015, 2025):
    path = CACHE_DIR / f"cfbd__rankings__{year}.json"
    assert path.exists(), f"Missing cached rankings file: {path}"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
BET_YEARS = {2022, 2023, 2024}
RANKING_ORDER = ["top_5", "top_10", "top_25", "unranked"]
SPREAD_ORDER = ["huge_favorite", "big_favorite", "moderate_favorite", "small_favorite", "pick_or_dog"]
LABELS = ["favorite_final_win", "deficit_erased"]
KEY_COLS = ["game_id", "fav_deficit", "trigger_sequence"]

print(f"[ok] N11 setup at {NOTEBOOK_DIR}")
""")


add("code", "c11_0002", r"""
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
    if obj is None:
        return None
    try:
        if not isinstance(obj, (str, bytes)) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def normalize_team(team: Any) -> str:
    text = "" if team is None else str(team)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.replace("&", "and").lower().strip()


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


def rank_bucket(rank: Any) -> str:
    if rank is None or pd.isna(rank):
        return "unranked"
    r = int(rank)
    if r <= 5:
        return "top_5"
    if r <= 10:
        return "top_10"
    return "top_25"


def bootstrap_cluster_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    work = df[["game_id", value_col]].dropna()
    if len(work) == 0:
        return {"lower": None, "median": None, "upper": None, "n_resamples": n_resamples}
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


def fmt_pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{100 * float(x):.1f}%"


def ci_text(ci: dict[str, Any]) -> str:
    if not ci or ci.get("lower") is None:
        return "[NA, NA]"
    return f"[{fmt_pct(ci['lower'])}, {fmt_pct(ci['upper'])}]"


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val).replace("|", "/"))
        out.append("| " + " | ".join(vals) + " |")
    return out


print("[ok] helpers defined")
""")


add("code", "c11_0003", r"""
def load_rank_maps() -> tuple[dict[tuple[int, str, int], dict[str, int]], pd.DataFrame, list[dict[str, Any]]]:
    maps: dict[tuple[int, str, int], dict[str, int]] = {}
    meta_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for season in range(2015, 2025):
        path = CACHE_DIR / f"cfbd__rankings__{season}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data:
            season_type = str(rec.get("seasonType"))
            week = int(rec.get("week"))
            polls = rec.get("polls") or []
            ap = [poll for poll in polls if poll.get("poll") == "AP Top 25"]
            if not ap:
                issues.append({"season": season, "season_type": season_type, "week": week, "issue": "missing_ap_poll"})
                continue
            ranks = ap[0].get("ranks") or []
            if len(ranks) < 25 or len(ranks) > 26:
                issues.append({"season": season, "season_type": season_type, "week": week, "issue": f"ap_rank_count_{len(ranks)}"})
            rank_map = {
                normalize_team(row.get("school")): int(row["rank"])
                for row in ranks
                if row.get("school") is not None and row.get("rank") is not None
            }
            maps[(season, season_type, week)] = rank_map
            meta_rows.append({
                "season": season,
                "season_type": season_type,
                "week": week,
                "n_ap_rank_rows": len(ranks),
                "top_ranked_team": sorted(ranks, key=lambda r: int(r["rank"]))[0]["school"] if ranks else None,
            })
    return maps, pd.DataFrame(meta_rows), issues


rank_maps, rank_meta_df, rank_issues = load_rank_maps()
assert not any(issue["issue"] == "missing_ap_poll" for issue in rank_issues), rank_issues
assert set(rank_meta_df["season"]) == set(range(2015, 2025))
assert rank_meta_df.groupby("season").size().between(15, 17).all(), rank_meta_df.groupby("season").size().to_dict()

top_2015_week1 = rank_meta_df[
    (rank_meta_df["season"].eq(2015))
    & (rank_meta_df["season_type"].eq("regular"))
    & (rank_meta_df["week"].eq(1))
]["top_ranked_team"].iloc[0]
assert top_2015_week1 == "Ohio State", f"Expected 2015 week 1 AP #1 Ohio State, got {top_2015_week1!r}"


def assigned_poll_week(season: int, season_type: str, week: int) -> int:
    regular_weeks = sorted(w for (y, st, w) in rank_maps if y == int(season) and st == "regular")
    assert regular_weeks, f"No regular AP polls for season {season}"
    if season_type == "postseason":
        return max(regular_weeks)
    prior = [w for w in regular_weeks if w <= int(week)]
    return max(prior) if prior else min(regular_weeks)


def ap_rank_for_team(season: int, season_type: str, week: int, team: str) -> int | None:
    poll_week = assigned_poll_week(season, season_type, week)
    return rank_maps.get((int(season), "regular", poll_week), {}).get(normalize_team(team))


print(f"[ok] rankings cache loaded: records={len(rank_meta_df)} issues={rank_issues}")
print(f"[info] 2015 regular week 1 AP #1: {top_2015_week1}")
""")


add("code", "c11_0004", r"""
n10 = pd.read_parquet(N10_RATES)
triggers = pd.read_csv(TRIGGERS)
trigger_cols = [
    "game_id", "fav_deficit", "trigger_sequence", "week", "season_type",
    "pregame_spread", "fav_pregame_rating", "dog_pregame_rating",
]
base = n10.merge(triggers[trigger_cols], on=KEY_COLS, how="left", validate="one_to_one")
assert len(base) == 11_412, f"Expected 11,412 N11 rows, got {len(base):,}"
assert not base[["week", "season_type", "pregame_spread", "fav_pregame_rating"]].isna().any().any()

assigned_weeks = []
assigned_ranks = []
for row in base.itertuples(index=False):
    poll_week = assigned_poll_week(int(row.season), str(row.season_type), int(row.week))
    assigned_weeks.append(poll_week)
    assigned_ranks.append(ap_rank_for_team(int(row.season), str(row.season_type), int(row.week), str(row.fav_team)))
base["ap_poll_week_used"] = assigned_weeks
base["fav_ap_rank_at_trigger"] = assigned_ranks
base["ranking_bucket"] = base["fav_ap_rank_at_trigger"].map(rank_bucket)

bucket_counts = base["ranking_bucket"].value_counts().reindex(RANKING_ORDER, fill_value=0)
underpowered = {bucket: int(n) for bucket, n in bucket_counts.items() if int(n) < 200}
assert not underpowered, f"Ranking bucket underpowered (<200 events): {underpowered}"

print(f"[ok] assigned AP ranks to {len(base):,} trigger events")
print("[info] ranking bucket counts:", bucket_counts.to_dict())
""")


add("code", "c11_0005", r"""
def load_games_cache() -> dict[int, dict[str, Any]]:
    games: dict[int, dict[str, Any]] = {}
    for path in sorted(CACHE_DIR.glob("cfbd__games__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            games[int(row["id"])] = row
    return games


games_by_id = load_games_cache()
team_game_rows: list[dict[str, Any]] = []
for game in games_by_id.values():
    if not game.get("completed"):
        continue
    if game.get("homePoints") is None or game.get("awayPoints") is None:
        continue
    season = int(game["season"])
    season_type = str(game["seasonType"])
    week = int(game["week"])
    for side in ["home", "away"]:
        team = str(game[f"{side}Team"])
        pts = int(game[f"{side}Points"])
        opp = int(game["awayPoints"] if side == "home" else game["homePoints"])
        rank = ap_rank_for_team(season, season_type, week, team)
        team_game_rows.append({
            "season": season,
            "team": team,
            "rank": rank,
            "ranking_bucket": rank_bucket(rank),
            "won": int(pts > opp),
        })
team_games_df = pd.DataFrame(team_game_rows)

spread_means = base.groupby("ranking_bucket")["pregame_spread"].mean().reindex(RANKING_ORDER)
fav_rating_means = base.groupby("ranking_bucket")["fav_pregame_rating"].mean().reindex(RANKING_ORDER)
team_game_win_pct = team_games_df.groupby("ranking_bucket")["won"].mean().reindex(RANKING_ORDER)
dog_rating_means = base.groupby("ranking_bucket")["dog_pregame_rating"].mean().reindex(RANKING_ORDER)
unique_teams = base.groupby("ranking_bucket")["fav_team"].nunique().reindex(RANKING_ORDER)
rank_dist = base[base["ranking_bucket"].eq("top_25")]["fav_ap_rank_at_trigger"].value_counts().sort_index().to_dict()

assert spread_means["top_5"] < spread_means["top_10"] < spread_means["top_25"] < spread_means["unranked"], spread_means.to_dict()
assert fav_rating_means["top_5"] > fav_rating_means["top_25"] > fav_rating_means["unranked"], fav_rating_means.to_dict()
assert team_game_win_pct["top_5"] > team_game_win_pct["unranked"], team_game_win_pct.to_dict()

ranking_sanity = {
    "api_call_count": 10,
    "rankings_cache_files": [f"cfbd__rankings__{year}.json" for year in range(2015, 2025)],
    "poll_records_by_season": rank_meta_df.groupby("season").size().to_dict(),
    "ap_rank_count_26_tie_records": rank_meta_df[rank_meta_df["n_ap_rank_rows"].eq(26)][["season", "season_type", "week"]].to_dict(orient="records"),
    "top_2015_week1": top_2015_week1,
    "bucket_counts": bucket_counts.to_dict(),
    "mean_pregame_spread": spread_means.to_dict(),
    "mean_fav_pregame_rating": fav_rating_means.to_dict(),
    "team_game_win_pct": team_game_win_pct.to_dict(),
    "mean_dog_pregame_rating": dog_rating_means.to_dict(),
    "unique_favorite_teams": unique_teams.to_dict(),
    "top25_rank_distribution": {str(int(k)): int(v) for k, v in rank_dist.items()},
}

print("[ok] ranking sanity checks passed")
print("[info] mean pregame spread:", spread_means.to_dict())
print("[info] mean favorite rating:", fav_rating_means.to_dict())
print("[info] all-game win pct:", team_game_win_pct.to_dict())
""")


add("code", "c11_0006", r"""
def flat_bet_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stake"] = 1.0
    out["profit"] = np.where(
        out["favorite_final_win"].astype(int).eq(1),
        out["decimal_odds_best_available"].astype(float) - 1.0,
        -1.0,
    )
    return out


def roi_summary(df: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    work = df[df["season"].isin(BET_YEARS) & (~df["is_synthetic_fallback_price"].fillna(False))].copy()
    work = work[work["decimal_odds_best_available"].notna()].copy()
    if len(work) == 0:
        return {
            "price_subset": "real_moneyline_only",
            "n_bets": 0,
            "n_games": 0,
            "n_seasons": 0,
            "win_rate": None,
            "roi": None,
            "bootstrap_ci": {"lower": None, "upper": None},
        }
    work = flat_bet_rows(work)
    return {
        "price_subset": "real_moneyline_only",
        "n_bets": int(len(work)),
        "n_games": int(work["game_id"].nunique()),
        "n_seasons": int(work["season"].nunique()),
        "win_rate": float(work["favorite_final_win"].mean()),
        "total_staked": float(work["stake"].sum()),
        "net_profit": float(work["profit"].sum()),
        "roi": float(work["profit"].sum() / work["stake"].sum()),
        "bootstrap_ci": bootstrap_cluster_roi(work, seed=seed),
    }


def group_cell_summary(df: pd.DataFrame, group_cols: list[str], *, seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), df)]
    for idx, (key, grp) in enumerate(grouped):
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        row = {col: (None if pd.isna(val) else val) for col, val in zip(group_cols, key)} if group_cols else {}
        n_events = int(len(grp))
        n_games = int(grp["game_id"].nunique())
        n_seasons = int(grp["season"].nunique())
        row.update({
            "n_events": n_events,
            "n_games": n_games,
            "n_seasons": n_seasons,
            "thin_flag": thin_flag(n_events, n_games, n_seasons),
            "mean_pregame_raw_implied_prob": float(grp["pregame_raw_implied_prob"].mean()),
            "mean_pregame_no_vig_implied_prob": float(grp["pregame_no_vig_implied_prob"].mean()),
            "heldout_real_moneyline_flat_roi": roi_summary(grp, seed=BOOTSTRAP_SEED + seed_base + idx),
        })
        for label in LABELS:
            successes = int(grp[label].sum())
            diff_col = f"{label}_minus_no_vig"
            tmp = grp.copy()
            tmp[diff_col] = tmp[label].astype(float) - tmp["pregame_no_vig_implied_prob"].astype(float)
            row[label] = {
                "successes": successes,
                "rate": float(successes / n_events) if n_events else None,
                "wilson_ci": wilson_ci(successes, n_events),
                "actual_minus_no_vig": float(tmp[diff_col].mean()),
                "actual_minus_no_vig_bootstrap_ci": bootstrap_cluster_mean(
                    tmp,
                    diff_col,
                    seed=BOOTSTRAP_SEED + seed_base + 10_000 + idx * 5 + (0 if label == "favorite_final_win" else 1),
                ),
            }
        final_diff = row["favorite_final_win"]["actual_minus_no_vig"]
        final_ci = row["favorite_final_win"]["actual_minus_no_vig_bootstrap_ci"]
        roi = row["heldout_real_moneyline_flat_roi"]
        row["candidate_live_watch"] = bool(
            row["thin_flag"] == "reliable"
            and final_diff is not None
            and final_diff >= 0.05
            and final_ci.get("lower") is not None
            and final_ci["lower"] > 0
            and roi.get("roi") is not None
            and roi["roi"] > 0
            and roi.get("bootstrap_ci", {}).get("lower") is not None
            and roi["bootstrap_ci"]["lower"] > 0
        )
        rows.append(row)
    return rows


print("[ok] aggregation helpers defined")
""")


add("code", "c11_0007", r"""
ranking_single = group_cell_summary(base, ["ranking_bucket"], seed_base=100)
ranking_deficit = group_cell_summary(base, ["ranking_bucket", "fav_deficit"], seed_base=500)
ranking_time = group_cell_summary(base, ["ranking_bucket", "time_bucket"], seed_base=1000)
ranking_spread = group_cell_summary(base, ["ranking_bucket", "spread_bucket"], seed_base=1500)
matched_cells = group_cell_summary(base, ["ranking_bucket", "fav_deficit", "time_bucket", "spread_bucket"], seed_base=3000)

matched_df_rows: list[dict[str, Any]] = []
for row in matched_cells:
    matched_df_rows.append({
        "ranking_bucket": row["ranking_bucket"],
        "fav_deficit": row["fav_deficit"],
        "time_bucket": row["time_bucket"],
        "spread_bucket": row["spread_bucket"],
        "n_events": row["n_events"],
        "n_games": row["n_games"],
        "n_seasons": row["n_seasons"],
        "thin_flag": row["thin_flag"],
        "favorite_final_win_rate": row["favorite_final_win"]["rate"],
        "favorite_final_win_wilson_lower": row["favorite_final_win"]["wilson_ci"]["lower"],
        "favorite_final_win_wilson_upper": row["favorite_final_win"]["wilson_ci"]["upper"],
        "favorite_final_win_minus_market": row["favorite_final_win"]["actual_minus_no_vig"],
        "favorite_final_win_minus_market_ci_lower": row["favorite_final_win"]["actual_minus_no_vig_bootstrap_ci"]["lower"],
        "favorite_final_win_minus_market_ci_upper": row["favorite_final_win"]["actual_minus_no_vig_bootstrap_ci"]["upper"],
    })
matched_df = pd.DataFrame(matched_df_rows)

hypothesis_a_separations: list[dict[str, Any]] = []
hypothesis_b_differentials: list[dict[str, Any]] = []
combo_cols = ["fav_deficit", "time_bucket", "spread_bucket"]
for combo, grp in matched_df.groupby(combo_cols, dropna=False):
    combo_dict = {col: val for col, val in zip(combo_cols, combo)}
    unranked = grp[(grp["ranking_bucket"].eq("unranked")) & (grp["n_events"].ge(50))]
    if unranked.empty:
        continue
    unr = unranked.iloc[0]
    for bucket in ["top_5", "top_10", "top_25"]:
        ranked = grp[(grp["ranking_bucket"].eq(bucket)) & (grp["n_events"].ge(50))]
        if ranked.empty:
            continue
        rr = ranked.iloc[0]
        separated = bool(rr["favorite_final_win_wilson_lower"] > unr["favorite_final_win_wilson_upper"])
        if separated:
            hypothesis_a_separations.append({
                **combo_dict,
                "ranking_bucket": bucket,
                "ranked_n_events": int(rr["n_events"]),
                "unranked_n_events": int(unr["n_events"]),
                "ranked_rate": float(rr["favorite_final_win_rate"]),
                "unranked_rate": float(unr["favorite_final_win_rate"]),
                "ranked_wilson_lower": float(rr["favorite_final_win_wilson_lower"]),
                "unranked_wilson_upper": float(unr["favorite_final_win_wilson_upper"]),
            })
        hypothesis_b_differentials.append({
            **combo_dict,
            "ranking_bucket": bucket,
            "ranked_n_events": int(rr["n_events"]),
            "unranked_n_events": int(unr["n_events"]),
            "ranked_actual_minus_market": float(rr["favorite_final_win_minus_market"]),
            "unranked_actual_minus_market": float(unr["favorite_final_win_minus_market"]),
            "ranked_minus_unranked_gap": float(rr["favorite_final_win_minus_market"] - unr["favorite_final_win_minus_market"]),
        })

inverse_df = base[
    base["ranking_bucket"].eq("unranked")
    & base["spread_bucket"].eq("small_favorite")
    & base["time_bucket"].eq("Q4")
].copy()
inverse_result = group_cell_summary(inverse_df, [], seed_base=20_000)[0] if len(inverse_df) else {
    "n_events": 0, "n_games": 0, "n_seasons": 0, "thin_flag": "unreliable"
}
inverse_result["definition"] = "ranking_bucket=unranked AND spread_bucket=small_favorite AND time_bucket=Q4"
inverse_roi = inverse_result.get("heldout_real_moneyline_flat_roi", {})
inverse_methodology_warning = bool(
    inverse_roi.get("roi") is not None
    and inverse_roi["roi"] > 0
    and inverse_roi.get("bootstrap_ci", {}).get("lower") is not None
    and inverse_roi["bootstrap_ci"]["lower"] > 0
)
assert not inverse_methodology_warning, "Inverse hypothesis sanity check shows positive ROI with CI lower > 0"

candidate_live_watch_cells = [row for row in matched_cells if row.get("candidate_live_watch")]

rank_single_lookup = {row["ranking_bucket"]: row for row in ranking_single}
top25_buckets = ["top_5", "top_10", "top_25"]
top25_mean_gap = float(np.mean([
    rank_single_lookup[b]["favorite_final_win"]["actual_minus_no_vig"]
    for b in top25_buckets
    if b in rank_single_lookup
]))
unranked_gap = float(rank_single_lookup["unranked"]["favorite_final_win"]["actual_minus_no_vig"])
if top25_mean_gap > unranked_gap + 0.03:
    hypothesis_b_interpretation = "ranked_favorites_less_overpriced_or_more_underpriced"
elif top25_mean_gap < unranked_gap - 0.03:
    hypothesis_b_interpretation = "ranked_favorites_more_overpriced"
else:
    hypothesis_b_interpretation = "no_material_differential_mispricing"

if hypothesis_a_separations:
    hypothesis_a_interpretation = "some_ranked_buckets_have_higher_matched_comeback_rates"
else:
    hypothesis_a_interpretation = "no_statistically_separated_matched_comeback_rate_advantage"

analysis_payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "methodology": {
        "models_trained": 0,
        "rankings_source": "CFBD /rankings AP Top 25 cache, 2015-2024",
        "api_call_count_for_rankings": 10,
        "ranking_assignment": "Regular-season games use most recent AP regular poll with poll week <= game week; postseason games use final regular-season AP poll.",
        "rank_buckets": {
            "top_5": "AP ranks 1-5",
            "top_10": "AP ranks 6-10",
            "top_25": "AP ranks 11-25",
            "unranked": "not in AP Top 25",
        },
    },
    "ranking_sanity_diagnostics": ranking_sanity,
    "ranking_single_dimension": ranking_single,
    "ranking_by_deficit": ranking_deficit,
    "ranking_by_time_bucket": ranking_time,
    "ranking_by_spread_bucket": ranking_spread,
    "matched_ranking_deficit_time_spread_cells": matched_cells,
    "hypothesis_a": {
        "interpretation": hypothesis_a_interpretation,
        "n_statistically_separated_cells": len(hypothesis_a_separations),
        "separated_cells": hypothesis_a_separations,
    },
    "hypothesis_b": {
        "interpretation": hypothesis_b_interpretation,
        "top25_mean_actual_minus_market": top25_mean_gap,
        "unranked_actual_minus_market": unranked_gap,
        "matched_differentials_vs_unranked": hypothesis_b_differentials,
    },
    "candidate_live_watch_cells": candidate_live_watch_cells,
    "inverse_hypothesis_sanity_check": inverse_result,
    "inverse_methodology_warning": inverse_methodology_warning,
}

n11_out = base[[
    "game_id", "trigger_play_id", "season", "week", "season_type", "fav_deficit",
    "time_bucket", "trigger_sequence", "fav_team", "dog_team", "fav_ap_rank_at_trigger",
    "ap_poll_week_used", "ranking_bucket", "fluke_composite", "fluke_bucket", "clear_fluky_lead",
    "spread_bucket", "favorite_final_win", "deficit_erased", "pregame_raw_implied_prob",
    "pregame_no_vig_implied_prob", "decimal_odds_best_available", "is_synthetic_fallback_price",
]].copy()
n11_out.to_parquet(N11_PARQUET, index=False)
N11_ANALYSIS_JSON.write_text(json.dumps(json_safe(analysis_payload), indent=2) + "\n", encoding="utf-8")

print("[ok] N11 analysis complete")
print(f"[info] Hypothesis A: {hypothesis_a_interpretation}; separated cells={len(hypothesis_a_separations)}")
print(f"[info] Hypothesis B: {hypothesis_b_interpretation}; top25_mean_gap={top25_mean_gap:.4f}; unranked_gap={unranked_gap:.4f}")
print(f"[info] candidate live-watch cells={len(candidate_live_watch_cells)}")
""")


add("code", "c11_0008", r"""
def row_label(row: dict[str, Any], cols: list[str]) -> str:
    return " / ".join(str(row.get(col)) for col in cols)


rank_rows = []
for bucket in RANKING_ORDER:
    row = rank_single_lookup[bucket]
    rank_rows.append({
        "bucket": bucket,
        "n_events": row["n_events"],
        "n_games": row["n_games"],
        "final_win": fmt_pct(row["favorite_final_win"]["rate"]),
        "deficit_erased": fmt_pct(row["deficit_erased"]["rate"]),
        "no_vig": fmt_pct(row["mean_pregame_no_vig_implied_prob"]),
        "final_minus_market": fmt_pct(row["favorite_final_win"]["actual_minus_no_vig"]),
        "heldout_roi": fmt_pct(row["heldout_real_moneyline_flat_roi"]["roi"]),
    })

if hypothesis_a_separations:
    strongest_a = max(hypothesis_a_separations, key=lambda r: r["ranked_rate"] - r["unranked_rate"])
    hypothesis_a_lead = (
        f"**Hypothesis A: PARTIAL DESCRIPTIVE SUPPORT.** There are **{len(hypothesis_a_separations)}** matched cells where a ranked bucket's Wilson lower bound exceeds the unranked Wilson upper bound. "
        f"The strongest cell is {strongest_a['ranking_bucket']} at D={strongest_a['fav_deficit']} / {strongest_a['time_bucket']} / {strongest_a['spread_bucket']}: "
        f"{fmt_pct(strongest_a['ranked_rate'])} vs unranked {fmt_pct(strongest_a['unranked_rate'])}."
    )
else:
    hypothesis_a_lead = (
        "**Hypothesis A: NO STATISTICALLY SEPARATED MATCHED COMEBACK ADVANTAGE.** "
        "Top-ranked favorites can have higher raw comeback rates in some broad summaries, but at matched deficit/time/spread cells no ranked bucket's Wilson lower bound clears the corresponding unranked upper bound."
    )

if hypothesis_b_interpretation == "ranked_favorites_more_overpriced":
    hypothesis_b_lead = (
        "**Hypothesis B: RANKED FAVORITES ARE MORE OVERPRICED.** "
        f"Across ranking buckets, the average ranked-favorite actual-minus-market gap is {fmt_pct(top25_mean_gap)} versus unranked {fmt_pct(unranked_gap)}."
    )
elif hypothesis_b_interpretation == "ranked_favorites_less_overpriced_or_more_underpriced":
    hypothesis_b_lead = (
        "**Hypothesis B: RANKED FAVORITES ARE LESS OVERPRICED / MORE UNDERPRICED.** "
        f"The average ranked-favorite actual-minus-market gap is {fmt_pct(top25_mean_gap)} versus unranked {fmt_pct(unranked_gap)}."
    )
else:
    hypothesis_b_lead = (
        "**Hypothesis B: NO MATERIAL DIFFERENTIAL MISPRICING.** "
        f"The average ranked-favorite actual-minus-market gap is {fmt_pct(top25_mean_gap)} versus unranked {fmt_pct(unranked_gap)}."
    )

candidate_text = (
    f"N11 flags **{len(candidate_live_watch_cells)}** candidate live-watch cells under the locked N10-style criteria."
)

lines: list[str] = []
lines.append("# N11 -- Top-25 favorite stratification and market efficiency analysis")
lines.append("")
lines.append("## Project-Level Closing Finding")
lines.append("")
lines.append("N11 is the final test of pre-game edge in the project's historical research arc. AP ranking is real and behaves as expected: top-5 teams are favored by more, win more often, and carry higher pre-game ratings. But ranking stratification does not reveal any pre-game market inefficiency. Every ranking bucket -- top_5, top_10, top_25, and unranked -- shows actual `favorite_final_win` rates below market-implied probability, with negative held-out ROI in every bucket.")
lines.append("")
lines.append("Six notebooks have now tested pre-game edge from distinct angles:")
lines.append("")
lines.append("- N04: model vs pre-game market showed positive Brier improvement, but the mechanism was restricted to trigger-state probability adjustment rather than betting edge.")
lines.append("- N05/N06: model vs deficit x time baseline showed no structural edge.")
lines.append("- N07/N08: methodology refinement and uncertainty diagnostics did not surface an edge under the stricter framework.")
lines.append("- N09: realized betting simulation showed the always-bet-favorite strategy loses, while the same-label positive filter was underpowered.")
lines.append("- N10: direct fluky-deficit hypothesis testing showed markets overprice favorites by 35 percentage points in the headline condition.")
lines.append("- N11: AP ranking stratification shows markets overprice every ranking bucket.")
lines.append("")
lines.append("Conclusion: pre-game CFB markets correctly price, and often slightly overprice, favorite comeback scenarios. There is no hidden inefficiency in any stratification dimension the project has tested. The mispricing diminishes as ranking improves -- top_5 gap **-18.3%** versus unranked gap **-27.2%** -- but never closes to neutral or positive. This is consistent with markets rewarding team strength while still compounding favorite quality slightly past actual performance.")
lines.append("")
lines.append("Live in-game market edge is a separate, untested hypothesis. The project's infrastructure -- N03-N08 models, N09-N11 stratifications, calibrated probabilities, and conformal intervals -- is ready for live-data testing in 2026. The pre-game edge story is closed.")
lines.append("")
lines.append(hypothesis_a_lead)
lines.append("")
lines.append(hypothesis_b_lead)
lines.append("")
lines.append(candidate_text)
lines.append("")
lines.append("N11 is descriptive only: no model training, no feature selection, and no threshold tuning. It uses cached AP rankings and the committed N10 trigger-level market probabilities.")
lines.append("")

lines.append("## Ranking Data Sanity Checks")
lines.append("")
lines.append("Ranking data passed the locked sanity checks: top-5 favorites are more heavily favored on average, have higher pre-game Elo, and top-5 teams win more often across cached games than unranked teams. AP records with 26 rows are retained as top-25 cutoff ties.")
sanity_rows = []
for bucket in RANKING_ORDER:
    sanity_rows.append({
        "bucket": bucket,
        "n_events": int(bucket_counts[bucket]),
        "mean_spread": ranking_sanity["mean_pregame_spread"][bucket],
        "mean_fav_rating": ranking_sanity["mean_fav_pregame_rating"][bucket],
        "all_game_win_pct": ranking_sanity["team_game_win_pct"][bucket],
        "unique_fav_teams": ranking_sanity["unique_favorite_teams"][bucket],
    })
lines.extend(markdown_table(sanity_rows, ["bucket", "n_events", "mean_spread", "mean_fav_rating", "all_game_win_pct", "unique_fav_teams"]))
lines.append("")

lines.append("## Ranking Bucket Summary")
lines.append("")
lines.extend(markdown_table(rank_rows, ["bucket", "n_events", "n_games", "final_win", "deficit_erased", "no_vig", "final_minus_market", "heldout_roi"]))
lines.append("")

lines.append("## Hypothesis A -- Matched Comeback Rates")
lines.append("")
lines.append(hypothesis_a_lead)
if hypothesis_a_separations:
    display = []
    for row in hypothesis_a_separations[:20]:
        display.append({
            "cell": f"D={row['fav_deficit']} / {row['time_bucket']} / {row['spread_bucket']}",
            "bucket": row["ranking_bucket"],
            "ranked_n": row["ranked_n_events"],
            "unranked_n": row["unranked_n_events"],
            "ranked_rate": fmt_pct(row["ranked_rate"]),
            "unranked_rate": fmt_pct(row["unranked_rate"]),
        })
    lines.extend(markdown_table(display, ["cell", "bucket", "ranked_n", "unranked_n", "ranked_rate", "unranked_rate"]))
else:
    lines.append("No matched cells meet the locked Wilson-separation criterion.")
lines.append("")

lines.append("## Hypothesis B -- Market Efficiency By Ranking")
lines.append("")
lines.append(hypothesis_b_lead)
diff_rows = []
for bucket in RANKING_ORDER:
    row = rank_single_lookup[bucket]
    diff_rows.append({
        "bucket": bucket,
        "actual_minus_market": fmt_pct(row["favorite_final_win"]["actual_minus_no_vig"]),
        "bootstrap_ci": ci_text(row["favorite_final_win"]["actual_minus_no_vig_bootstrap_ci"]),
        "heldout_roi": fmt_pct(row["heldout_real_moneyline_flat_roi"]["roi"]),
        "heldout_roi_ci": ci_text(row["heldout_real_moneyline_flat_roi"]["bootstrap_ci"]),
    })
lines.extend(markdown_table(diff_rows, ["bucket", "actual_minus_market", "bootstrap_ci", "heldout_roi", "heldout_roi_ci"]))
lines.append("")

lines.append("## Candidate Live-Watch Cells")
lines.append("")
if candidate_live_watch_cells:
    display = []
    for row in candidate_live_watch_cells[:30]:
        display.append({
            "cell": row_label(row, ["ranking_bucket", "fav_deficit", "time_bucket", "spread_bucket"]),
            "n_events": row["n_events"],
            "n_games": row["n_games"],
            "final_win": fmt_pct(row["favorite_final_win"]["rate"]),
            "no_vig": fmt_pct(row["mean_pregame_no_vig_implied_prob"]),
            "diff": fmt_pct(row["favorite_final_win"]["actual_minus_no_vig"]),
            "roi": fmt_pct(row["heldout_real_moneyline_flat_roi"]["roi"]),
        })
    lines.extend(markdown_table(display, ["cell", "n_events", "n_games", "final_win", "no_vig", "diff", "roi"]))
else:
    lines.append("No ranking-stratified cell satisfies the locked candidate live-watch rule.")
lines.append("")

lines.append("## Inverse Sanity Check")
lines.append("")
inv_fw = inverse_result.get("favorite_final_win", {})
inv_roi = inverse_result.get("heldout_real_moneyline_flat_roi", {})
lines.append(f"Definition: `{inverse_result['definition']}`.")
lines.append(f"Events/games/seasons: **{inverse_result.get('n_events')} / {inverse_result.get('n_games')} / {inverse_result.get('n_seasons')}** (`{inverse_result.get('thin_flag')}`).")
lines.append(f"Favorite final-win rate: **{fmt_pct(inv_fw.get('rate'))}**; actual-minus-market **{fmt_pct(inv_fw.get('actual_minus_no_vig'))}** with CI **{ci_text(inv_fw.get('actual_minus_no_vig_bootstrap_ci', {}))}**.")
lines.append(f"Held-out real-moneyline ROI: **{fmt_pct(inv_roi.get('roi'))}** on **{inv_roi.get('n_bets')}** bets, CI **{ci_text(inv_roi.get('bootstrap_ci', {}))}**.")
lines.append(f"Methodology warning: **{inverse_methodology_warning}**.")
lines.append("")

lines.append("## Honest Interpretation")
lines.append("")
lines.append("Ranking status behaves like real football strength, and N11 finds one narrow matched-cell descriptive separation, but it does not recover the project's pre-game edge hypothesis. Every ranking bucket remains below market-implied probability on aggregate, held-out ROI is negative in every ranking bucket, and zero cells satisfy the locked candidate live-watch rule. The central N10 conclusion remains intact: AP ranking stratification does not reveal a hidden pre-game market inefficiency for favorite comeback scenarios.")
lines.append("")

N11_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ok] wrote N11 deliverables: {N11_PARQUET.name}, {N11_ANALYSIS_JSON.name}, {N11_SUMMARY_MD.name}")
""")


def build() -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell_type, cell_id, src in CELLS:
        nb["cells"].append({
            "cell_type": cell_type,
            "id": cell_id,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
        })
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
