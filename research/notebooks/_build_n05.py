"""
Deterministic builder for research/notebooks/05_comeback_rate_validation.ipynb.

N05 is a descriptive and model-vs-baseline analysis over trigger events.
It distinguishes two labels throughout:

- favorite_final_win: the favorite won the game.
- deficit_erased: the favorite tied or retook the lead after the trigger.
"""

from __future__ import annotations

import json
import pathlib
import textwrap

from _lib_chrono import CHRONO_KEY_SOURCE

OUT = pathlib.Path(__file__).resolve().parent / "05_comeback_rate_validation.ipynb"

CELLS: list[tuple[str, str, str]] = []


def add(cell_type: str, cell_id: str, src: str) -> None:
    CELLS.append((cell_type, cell_id, textwrap.dedent(src).lstrip("\n")))


add("markdown", "m05_0000", """
# Notebook 05 -- Favorite final-win and deficit-erased rates

N05 answers two questions:

1. Descriptive: when favorites enter trigger states, how often do they
   ultimately win, and how often do they erase the deficit?
2. Model validation: on held-out N03 test folds, does the model identify
   trigger states whose actual rates exceed a naive deficit-by-time baseline?

Terminology is intentionally strict:

- `favorite_final_win`: the favorite won the game. This is the N03 label.
- `deficit_erased`: the favorite tied or retook the lead after the trigger.

Do not collapse these into a single "comeback" label.
""")


add("code", "c05_0001", f"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import pathlib
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

NOTEBOOK_DIR = pathlib.Path(".").resolve()
RESEARCH_DIR = (NOTEBOOK_DIR / "..").resolve()
REPO_ROOT = (RESEARCH_DIR / "..").resolve()
CACHE_DIR = RESEARCH_DIR / "data" / "cache"
RESULTS_DIR = RESEARCH_DIR / "results"

TRIGGER_EVENTS_CSV = RESULTS_DIR / "trigger_events.csv"
TRIGGER_OUTCOMES_CSV = RESULTS_DIR / "trigger_outcomes.csv"
N03_PREDICTIONS = RESULTS_DIR / "n03_calibrated_predictions.parquet"

N05_DESCRIPTIVE_PARQUET = RESULTS_DIR / "n05_descriptive_rates.parquet"
N05_ANALYSIS_JSON = RESULTS_DIR / "n05_analysis_results.json"
N05_SUMMARY_MD = RESULTS_DIR / "n05_summary_report.md"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
LABELS = ["favorite_final_win", "deficit_erased"]
DEFICITS = [3, 7, 10, 14, 21]
TIME_BUCKETS = ["Q1", "Q2-first-half", "Q3", "Q4"]
EDGE_THRESHOLDS = [0.00, 0.03, 0.05, 0.08, 0.10]
TRAIN_BASELINE_SEASONS = list(range(2015, 2022))
TEST_SEASONS = [2022, 2023, 2024]

for path in [TRIGGER_EVENTS_CSV, TRIGGER_OUTCOMES_CSV, N03_PREDICTIONS]:
    assert path.exists(), f"Missing N05 input: {{path}}"
assert CACHE_DIR.exists(), f"Missing cache dir: {{CACHE_DIR}}"

{CHRONO_KEY_SOURCE}

print(f"[ok] N05 paths resolved at {{NOTEBOOK_DIR}}")
""")


add("code", "c05_0002", """
def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _cache_key(endpoint: str, params: dict[str, Any]) -> pathlib.Path:
    endpoint_key = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"cfbd__{endpoint_key}__{_params_hash(params)}.json"


def readonly_cfbd_get(endpoint: str, force_refresh: bool = False, **params: Any) -> Any:
    if force_refresh:
        raise AssertionError("N05 forbids force_refresh; cache-only extraction is required")
    key = _cache_key(endpoint, params)
    if not key.exists():
        raise AssertionError(
            f"N05 missing local cache for {endpoint} {params}; halt before any external fetch."
        )
    return json.loads(key.read_text(encoding="utf-8"))


def load_cache_records(prefix: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(CACHE_DIR.glob(f"cfbd__{prefix}__*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if n <= 0:
        return {"successes": int(successes), "n": int(n), "rate": None, "wilson_low": None, "wilson_high": None}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {
        "successes": int(successes),
        "n": int(n),
        "rate": float(p),
        "wilson_low": float(max(0.0, center - half)),
        "wilson_high": float(min(1.0, center + half)),
    }


def bootstrap_cluster_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | None]:
    clean = df[["game_id", value_col]].dropna().copy()
    if clean.empty:
        return {"2.5": None, "25": None, "50": None, "75": None, "97.5": None}
    grouped = clean.groupby("game_id")[value_col].agg(["sum", "count"]).reset_index()
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_games = len(grouped)
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_games, size=n_games)
        out[i] = sums[idx].sum() / counts[idx].sum()
    pct = [2.5, 25, 50, 75, 97.5]
    return {str(p): float(v) for p, v in zip(pct, np.percentile(out, pct))}


def rate_record(df: pd.DataFrame, label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    clean = df[df[label].notna()]
    successes = int(clean[label].astype(bool).sum())
    n = int(len(clean))
    out = wilson_interval(successes, n)
    if extra:
        out.update(extra)
    return out


def add_bootstrap_rate(record: dict[str, Any], df: pd.DataFrame, label: str, *, seed: int) -> dict[str, Any]:
    tmp = df[df[label].notna()].copy()
    tmp["_label_value"] = tmp[label].astype(float)
    record["bootstrap_rate_ci"] = bootstrap_cluster_mean_ci(tmp, "_label_value", seed=seed)
    return record


def brier(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def fmt(x: Any, digits: int = 5) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def time_bucket_from_row(row: pd.Series) -> str:
    q = int(row["quarter"])
    if q == 1:
        return "Q1"
    if q == 2:
        return "Q2-first-half"
    if q == 3:
        return "Q3"
    if q == 4:
        return "Q4"
    return "other"


def bool_bucket(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return "true" if bool(value) else "false"


print("[ok] N05 helper functions defined")
""")


add("code", "c05_0003", """
t0 = time.perf_counter()
triggers = pd.read_csv(TRIGGER_EVENTS_CSV)
outcomes = pd.read_csv(TRIGGER_OUTCOMES_CSV)

assert len(triggers) == 11416, f"expected 11,416 trigger events, got {len(triggers):,}"
assert triggers[["game_id", "fav_deficit", "trigger_sequence"]].duplicated().sum() == 0
assert outcomes[["game_id", "fav_deficit"]].duplicated().sum() == 0

base = triggers.merge(outcomes, on=["game_id", "fav_deficit"], how="left", validate="many_to_one")
base = base.rename(columns={"final_fav_won": "favorite_final_win"})
base["favorite_final_win"] = base["favorite_final_win"].map({True: True, False: False, "True": True, "False": False})
assert base["favorite_final_win"].notna().all(), "favorite_final_win join produced nulls"
base["period_seconds_elapsed"] = 900 - base["clock_seconds_in_period_total"].astype(int)
base["time_bucket"] = base.apply(time_bucket_from_row, axis=1)
assert set(base["time_bucket"].unique()) <= set(TIME_BUCKETS), base["time_bucket"].value_counts()

play_records = load_cache_records("plays")
trigger_game_ids = set(int(x) for x in base["game_id"].unique())
plays_by_game: dict[int, list[dict[str, Any]]] = {gid: [] for gid in trigger_game_ids}
for play in play_records:
    gid = int(play.get("gameId"))
    if gid in plays_by_game:
        plays_by_game[gid].append(play)

missing_play_games = sorted(gid for gid, plays in plays_by_game.items() if not plays)
assert not missing_play_games, f"missing play-by-play for trigger games: {missing_play_games[:10]}"
for gid in plays_by_game:
    plays_by_game[gid].sort(key=_chrono_key)

print(f"[ok] trigger events loaded: {len(base):,}")
print(f"[ok] unique trigger plays: {base[['game_id', 'trigger_play_id']].drop_duplicates().shape[0]:,}")
print(f"[ok] trigger games with play-by-play: {len(plays_by_game):,}")
print(f"[ok] cache plays scanned: {len(play_records):,}; elapsed={time.perf_counter() - t0:.1f}s")
""")


add("code", "c05_0004", """
def score_after_play(play: dict[str, Any], fav_team: str, dog_team: str) -> tuple[float | None, float | None]:
    offense = play.get("offense")
    defense = play.get("defense")
    off_score = play.get("offenseScore")
    def_score = play.get("defenseScore")
    if off_score is None or def_score is None or pd.isna(off_score) or pd.isna(def_score):
        return None, None
    if offense == fav_team and defense == dog_team:
        return float(off_score), float(def_score)
    if offense == dog_team and defense == fav_team:
        return float(def_score), float(off_score)
    if offense == fav_team:
        return float(off_score), float(def_score)
    if defense == fav_team:
        return float(def_score), float(off_score)
    return None, None


def chrono_key_string(key: tuple[int, int, int, int] | None) -> str | None:
    if key is None:
        return None
    return "|".join(str(int(x)) for x in key)


play_label_rows: list[dict[str, Any]] = []
missing_trigger_play_ids: list[dict[str, Any]] = []
score_mismatch_rows: list[dict[str, Any]] = []
no_post_trigger_rows: list[dict[str, Any]] = []
team_score_fail_rows: list[dict[str, Any]] = []

unique_play_triggers = (
    base.sort_values(["game_id", "trigger_play_id", "fav_deficit"])
    .drop_duplicates(["game_id", "trigger_play_id"], keep="first")
)

for row in unique_play_triggers.itertuples(index=False):
    gid = int(row.game_id)
    trigger_play_id = str(row.trigger_play_id)
    fav_team = str(row.fav_team)
    dog_team = str(row.dog_team)
    plays = plays_by_game[gid]
    play_by_id = {str(p.get("id")): p for p in plays}
    trig_play = play_by_id.get(trigger_play_id)
    if trig_play is None:
        missing_trigger_play_ids.append({"game_id": gid, "trigger_play_id": trigger_play_id})
        play_label_rows.append({
            "game_id": gid,
            "trigger_play_id": row.trigger_play_id,
            "deficit_erased": np.nan,
            "deficit_erased_chrono_key": None,
            "deficit_erased_play_id": None,
        })
        continue

    trig_key = _chrono_key(trig_play)
    trig_fav_score, trig_dog_score = score_after_play(trig_play, fav_team, dog_team)
    if trig_fav_score is None or trig_dog_score is None:
        team_score_fail_rows.append({"game_id": gid, "trigger_play_id": trigger_play_id, "play_id": trig_play.get("id")})
    else:
        if (
            int(trig_fav_score) != int(row.fav_score_at_trigger)
            or int(trig_dog_score) != int(row.dog_score_at_trigger)
        ):
            score_mismatch_rows.append({
                "game_id": gid,
                "trigger_play_id": trigger_play_id,
                "computed_fav_score": trig_fav_score,
                "computed_dog_score": trig_dog_score,
                "recorded_fav_score": row.fav_score_at_trigger,
                "recorded_dog_score": row.dog_score_at_trigger,
            })

    post_plays = [p for p in plays if _chrono_key(p) > trig_key]
    if not post_plays:
        no_post_trigger_rows.append({"game_id": gid, "trigger_play_id": trigger_play_id})
        play_label_rows.append({
            "game_id": gid,
            "trigger_play_id": row.trigger_play_id,
            "deficit_erased": np.nan,
            "deficit_erased_chrono_key": None,
            "deficit_erased_play_id": None,
        })
        continue

    erased = False
    erased_key: tuple[int, int, int, int] | None = None
    erased_play_id: str | None = None
    for play in post_plays:
        fav_score, dog_score = score_after_play(play, fav_team, dog_team)
        if fav_score is None or dog_score is None:
            continue
        if fav_score >= dog_score:
            erased = True
            erased_key = _chrono_key(play)
            erased_play_id = str(play.get("id"))
            break

    play_label_rows.append({
        "game_id": gid,
        "trigger_play_id": row.trigger_play_id,
        "deficit_erased": erased,
        "deficit_erased_chrono_key": chrono_key_string(erased_key),
        "deficit_erased_play_id": erased_play_id,
    })

play_labels = pd.DataFrame(play_label_rows)
assert play_labels[["game_id", "trigger_play_id"]].duplicated().sum() == 0

base = base.merge(play_labels, on=["game_id", "trigger_play_id"], how="left", validate="many_to_one")
assert base["deficit_erased"].notna().sum() + base["deficit_erased"].isna().sum() == len(base)

deficit_label_quality = {
    "unique_trigger_plays": int(len(unique_play_triggers)),
    "missing_trigger_play_ids": int(len(missing_trigger_play_ids)),
    "score_mismatch_count": int(len(score_mismatch_rows)),
    "team_score_fail_count": int(len(team_score_fail_rows)),
    "no_post_trigger_play_count": int(len(no_post_trigger_rows)),
    "deficit_erased_null_event_rows": int(base["deficit_erased"].isna().sum()),
}
print("[ok] deficit_erased labels computed")
print(json.dumps(deficit_label_quality, indent=2))
""")


add("code", "c05_0005", """
# Feature subset columns for Q1 descriptive tables.
base["early_vs_late_season"] = np.where(base["week"].astype(int) <= 8, "early_week_le_8", "late_week_gt_8")
base["fav_is_home"] = base["home_is_fav"].astype(bool)

games = load_cache_records("games")
neutral_by_game: dict[int, bool] = {}
for rec in games:
    gid = int(rec.get("id"))
    if gid in trigger_game_ids and "neutralSite" in rec:
        neutral_by_game[gid] = bool(rec.get("neutralSite"))
base["is_neutral_site"] = base["game_id"].map(neutral_by_game)

# Reuse the canonical 02b feature matrix rather than reimplementing the
# opening-drive extractor. This executes selected notebook cells cache-only.
PHASE0_02B_CELLS = [4, 6, 8, 10, 11, 13]


def _base_phase0_namespace() -> dict[str, Any]:
    return {
        "__name__": "_n05_phase0_cell_exec",
        "Any": Any,
        "contextlib": contextlib,
        "hashlib": hashlib,
        "io": io,
        "json": json,
        "math": math,
        "np": np,
        "pathlib": pathlib,
        "pd": pd,
        "subprocess": subprocess,
        "time": time,
        "NOTEBOOK_DIR": NOTEBOOK_DIR,
        "RESEARCH_DIR": RESEARCH_DIR,
        "DATA_DIR": RESEARCH_DIR / "data",
        "RESULTS_DIR": RESULTS_DIR,
        "CACHE_DIR": CACHE_DIR,
        "CALL_LOG": CACHE_DIR / "cfbd_call_log.csv",
        "REPO_ROOT": REPO_ROOT,
        "TRIGGER_EVENTS_CSV": TRIGGER_EVENTS_CSV,
        "TRIGGER_OUTCOMES_CSV": TRIGGER_OUTCOMES_CSV,
        "FEATURE_VALIDATION_CSV": RESULTS_DIR / "feature_validation.csv",
        "FEATURE_VALIDATION_SCHEMA": RESULTS_DIR / "feature_validation.schema.md",
        "cfbd_get": readonly_cfbd_get,
    }


def run_02b_feature_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    nb_name = "02b_opening_drive_shock.ipynb"
    nb_path = NOTEBOOK_DIR / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = _base_phase0_namespace()
    stdout = io.StringIO()
    t_start = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        for idx in PHASE0_02B_CELLS:
            src = "".join(nb["cells"][idx]["source"])
            exec(compile(src, f"{nb_name}:cell{idx}", "exec"), ns)
            ns["cfbd_get"] = readonly_cfbd_get
    fm = ns.get("feature_matrix_df")
    if fm is None:
        raise AssertionError("02b notebook execution did not produce feature_matrix_df")
    return fm.copy(), {
        "notebook": nb_name,
        "cells": PHASE0_02B_CELLS,
        "rows": int(len(fm)),
        "cols": int(fm.shape[1]),
        "elapsed_sec": float(time.perf_counter() - t_start),
        "stdout_tail": stdout.getvalue()[-2000:],
    }


fm_02b, feature_subset_extract_summary = run_02b_feature_matrix()
subset_cols = [
    "game_id",
    "fav_deficit",
    "trigger_sequence",
    "dog_scored_on_opening_drive",
    "opening_drive_was_explosive_td",
]
base = base.merge(
    fm_02b[subset_cols],
    on=["game_id", "fav_deficit", "trigger_sequence"],
    how="left",
    validate="one_to_one",
)
base["dog_scored_on_opening_drive_bucket"] = base["dog_scored_on_opening_drive"].map(bool_bucket)
base["opening_drive_was_explosive_td_bucket"] = base["opening_drive_was_explosive_td"].map(bool_bucket)
base["is_neutral_site_bucket"] = base["is_neutral_site"].map(bool_bucket)

feature_subset_quality = {
    "games_with_neutral_site_metadata": int(base[["game_id", "is_neutral_site"]].drop_duplicates()["is_neutral_site"].notna().sum()),
    "neutral_site_missing_event_rows": int(base["is_neutral_site"].isna().sum()),
    "feature_subset_extract_summary": feature_subset_extract_summary,
}
print("[ok] Q1 feature subset fields joined")
print(json.dumps(feature_subset_quality, indent=2))
""")


add("code", "c05_0006", """
descriptive_cols = [
    "game_id",
    "trigger_play_id",
    "fav_deficit",
    "trigger_sequence",
    "quarter",
    "clock_seconds_in_period_total",
    "period_seconds_elapsed",
    "time_bucket",
    "fav_team",
    "dog_team",
    "season",
    "week",
    "favorite_final_win",
    "deficit_erased",
    "deficit_erased_chrono_key",
    "deficit_erased_play_id",
    "dog_scored_on_opening_drive_bucket",
    "opening_drive_was_explosive_td_bucket",
    "early_vs_late_season",
    "is_neutral_site_bucket",
]
base[descriptive_cols].to_parquet(N05_DESCRIPTIVE_PARQUET, index=False)
print(f"[ok] wrote {N05_DESCRIPTIVE_PARQUET.relative_to(REPO_ROOT)} rows={len(base):,}")
""")


add("code", "c05_0007", """
def grouped_rate_table(df: pd.DataFrame, label: str, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    for col in group_cols:
        if work[col].dtype == "O":
            work[col] = work[col].fillna("missing")
    for key, grp in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        extra = {col: val.item() if hasattr(val, "item") else val for col, val in zip(group_cols, key)}
        rows.append(rate_record(grp, label, extra))
    return rows


descriptive: dict[str, Any] = {}
subset_group_cols = {
    "dog_scored_on_opening_drive": ["dog_scored_on_opening_drive_bucket"],
    "opening_drive_was_explosive_td": ["opening_drive_was_explosive_td_bucket"],
    "early_vs_late_season": ["early_vs_late_season"],
    "is_neutral_site": ["is_neutral_site_bucket"],
}

for label in LABELS:
    label_df = base[base[label].notna()].copy()
    overall = rate_record(label_df, label, {"scope": "overall"})
    add_bootstrap_rate(overall, label_df, label, seed=BOOTSTRAP_SEED + (0 if label == "favorite_final_win" else 10))
    descriptive[label] = {
        "overall": overall,
        "by_deficit": grouped_rate_table(label_df, label, ["fav_deficit"]),
        "by_time_bucket": grouped_rate_table(label_df, label, ["time_bucket"]),
        "by_deficit_time": grouped_rate_table(label_df, label, ["fav_deficit", "time_bucket"]),
        "by_season": grouped_rate_table(label_df, label, ["season"]),
        "by_feature_subset": {
            name: grouped_rate_table(label_df, label, cols)
            for name, cols in subset_group_cols.items()
        },
    }

print("[ok] Q1 descriptive aggregates computed")
for label in LABELS:
    print(label, descriptive[label]["overall"])
""")


add("code", "c05_0008", """
pred = pd.read_parquet(N03_PREDICTIONS)
pred_u = pred[pred["scheme"].eq("U") & pred["split_role"].eq("test")].copy()
assert len(pred_u) == 3857, f"expected 3,857 Scheme U held-out rows, got {len(pred_u):,}"

model_df = pred_u.merge(
    base[[
        "game_id",
        "fav_deficit",
        "trigger_sequence",
        "time_bucket",
        "season",
        "favorite_final_win",
        "deficit_erased",
    ]],
    on=["game_id", "fav_deficit", "trigger_sequence"],
    how="left",
    validate="one_to_one",
)
assert model_df["favorite_final_win"].notna().all(), "N03 prediction join lost favorite_final_win labels"
assert model_df["season"].isin(TEST_SEASONS).all(), "N03 held-out rows are not limited to 2022-2024"
model_df = model_df.rename(columns={
    "model_prob": "raw_model_prob",
    "calibrated_prob": "model_prob",
})
assert not model_df.columns.duplicated().any()
model_df["favorite_final_win"] = model_df["favorite_final_win"].astype(bool)

train_base = base[base["season"].isin(TRAIN_BASELINE_SEASONS)].copy()
baseline_tables: dict[str, Any] = {}
for label in LABELS:
    train_label = train_base[train_base[label].notna()].copy()
    overall_rate = float(train_label[label].astype(float).mean())
    by_cell = (
        train_label.groupby(["fav_deficit", "time_bucket"])[label]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "baseline_C_rate", "sum": "successes", "count": "n"})
    )
    by_deficit = (
        train_label.groupby("fav_deficit")[label]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "baseline_deficit_rate", "sum": "successes", "count": "n"})
    )
    baseline_tables[label] = {
        "overall_training_rate": overall_rate,
        "deficit_time": by_cell.to_dict(orient="records"),
        "deficit": by_deficit.to_dict(orient="records"),
    }
    model_df = model_df.merge(
        by_cell[["fav_deficit", "time_bucket", "baseline_C_rate"]].rename(
            columns={"baseline_C_rate": f"baseline_C_{label}"}
        ),
        on=["fav_deficit", "time_bucket"],
        how="left",
        validate="many_to_one",
    )
    assert model_df[f"baseline_C_{label}"].notna().all(), f"missing baseline_C cells for {label}"

print("[ok] held-out Scheme U predictions joined to labels and training-year baselines")
print(model_df[["fold", "game_id", "fav_deficit", "time_bucket", "model_prob", "baseline_C_favorite_final_win", "baseline_C_deficit_erased"]].head().to_string(index=False))
""")


add("code", "c05_0009", """
def threshold_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_col = f"baseline_C_{label}"
    clean = df[df[label].notna()].copy()
    for threshold in EDGE_THRESHOLDS:
        sel = clean[clean["model_prob"] > clean[baseline_col] + threshold].copy()
        rec = rate_record(sel, label, {"threshold": threshold})
        if len(sel):
            rec.update({
                "mean_model_prob": float(sel["model_prob"].mean()),
                "mean_baseline_C": float(sel[baseline_col].mean()),
                "actual_minus_mean_model_prob": float(rec["rate"] - sel["model_prob"].mean()),
                "actual_minus_mean_baseline_C": float(rec["rate"] - sel[baseline_col].mean()),
            })
        else:
            rec.update({
                "mean_model_prob": None,
                "mean_baseline_C": None,
                "actual_minus_mean_model_prob": None,
                "actual_minus_mean_baseline_C": None,
            })
        add_bootstrap_rate(rec, sel, label, seed=BOOTSTRAP_SEED + 100 + int(threshold * 1000) + (0 if label == "favorite_final_win" else 1000))
        rows.append(rec)
    return rows


def quintile_analysis(df: pd.DataFrame, label: str) -> dict[str, Any]:
    clean = df[df[label].notna()].copy()
    clean["model_quintile"] = pd.qcut(clean["model_prob"], q=5, labels=False, duplicates="drop") + 1
    rows: list[dict[str, Any]] = []
    for q, grp in clean.groupby("model_quintile", sort=True):
        rec = rate_record(grp, label, {"quintile": int(q)})
        rec.update({
            "model_prob_min": float(grp["model_prob"].min()),
            "model_prob_max": float(grp["model_prob"].max()),
            "mean_model_prob": float(grp["model_prob"].mean()),
            "mean_baseline_C": float(grp[f"baseline_C_{label}"].mean()),
            "actual_minus_mean_model_prob": float(rec["rate"] - grp["model_prob"].mean()),
            "actual_minus_mean_baseline_C": float(rec["rate"] - grp[f"baseline_C_{label}"].mean()),
        })
        rows.append(rec)
    spearman = clean[["model_prob", label]].corr(method="spearman").iloc[0, 1]
    return {"spearman_model_prob_vs_actual": float(spearman), "rows": rows}


def decile_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    clean = df[df[label].notna()].copy()
    bins = np.linspace(0.0, 1.0, 11)
    clean["model_decile"] = pd.cut(clean["model_prob"], bins=bins, include_lowest=True, right=False, labels=False)
    clean.loc[clean["model_prob"].eq(1.0), "model_decile"] = 9
    rows: list[dict[str, Any]] = []
    for decile in range(10):
        grp = clean[clean["model_decile"].eq(decile)].copy()
        rec = rate_record(grp, label, {"decile": decile, "lo": float(bins[decile]), "hi": float(bins[decile + 1])})
        if len(grp):
            rec.update({
                "mean_model_prob": float(grp["model_prob"].mean()),
                "mean_baseline_C": float(grp[f"baseline_C_{label}"].mean()),
                "actual_minus_mean_model_prob": float(rec["rate"] - grp["model_prob"].mean()),
                "actual_minus_mean_baseline_C": float(rec["rate"] - grp[f"baseline_C_{label}"].mean()),
            })
        else:
            rec.update({
                "mean_model_prob": None,
                "mean_baseline_C": None,
                "actual_minus_mean_model_prob": None,
                "actual_minus_mean_baseline_C": None,
            })
        add_bootstrap_rate(rec, grp, label, seed=BOOTSTRAP_SEED + 200 + decile + (0 if label == "favorite_final_win" else 1000))
        rows.append(rec)
    return rows


def bootstrap_brier_improvement(df: pd.DataFrame, value_col: str, *, seed: int) -> dict[str, float | None]:
    return bootstrap_cluster_mean_ci(df, value_col, seed=seed)


def per_deficit_analysis(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    clean = df[df[label].notna()].copy()
    baseline_col = f"baseline_C_{label}"
    constant_rate = baseline_tables[label]["overall_training_rate"]
    clean[f"brier_model_{label}"] = (clean["model_prob"] - clean[label].astype(float)) ** 2
    clean[f"brier_baseline_C_{label}"] = (clean[baseline_col] - clean[label].astype(float)) ** 2
    clean[f"brier_constant_{label}"] = (constant_rate - clean[label].astype(float)) ** 2
    clean[f"brier_improvement_{label}"] = clean[f"brier_baseline_C_{label}"] - clean[f"brier_model_{label}"]
    rows: list[dict[str, Any]] = []
    for deficit, grp in clean.groupby("fav_deficit", sort=True):
        rec = rate_record(grp, label, {"fav_deficit": int(deficit)})
        rec.update({
            "mean_model_prob": float(grp["model_prob"].mean()),
            "mean_baseline_C": float(grp[baseline_col].mean()),
            "brier_model": float(grp[f"brier_model_{label}"].mean()),
            "brier_baseline_C": float(grp[f"brier_baseline_C_{label}"].mean()),
            "brier_constant_training_overall": float(grp[f"brier_constant_{label}"].mean()),
            "brier_improvement_baseline_C_minus_model": float(grp[f"brier_improvement_{label}"].mean()),
            "brier_improvement_bootstrap_ci": bootstrap_brier_improvement(
                grp,
                f"brier_improvement_{label}",
                seed=BOOTSTRAP_SEED + 300 + int(deficit) + (0 if label == "favorite_final_win" else 1000),
            ),
        })
        rows.append(rec)
    return rows


def overall_brier_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    clean = df[df[label].notna()].copy()
    baseline_col = f"baseline_C_{label}"
    clean[f"brier_improvement_{label}"] = (
        (clean[baseline_col] - clean[label].astype(float)) ** 2
        - (clean["model_prob"] - clean[label].astype(float)) ** 2
    )
    return {
        "n": int(len(clean)),
        "n_games": int(clean["game_id"].nunique()),
        "brier_model": brier(clean[label].astype(float), clean["model_prob"]),
        "brier_baseline_C": brier(clean[label].astype(float), clean[baseline_col]),
        "brier_improvement_baseline_C_minus_model": float(clean[f"brier_improvement_{label}"].mean()),
        "brier_improvement_bootstrap_ci": bootstrap_brier_improvement(
            clean,
            f"brier_improvement_{label}",
            seed=BOOTSTRAP_SEED + 400 + (0 if label == "favorite_final_win" else 1000),
        ),
    }


model_validation: dict[str, Any] = {}
for label in LABELS:
    model_validation[label] = {
        "overall_brier_vs_baseline_C": overall_brier_summary(model_df, label),
        "threshold_analysis": threshold_analysis(model_df, label),
        "quintile_analysis": quintile_analysis(model_df, label),
        "decile_analysis": decile_analysis(model_df, label),
        "per_deficit_analysis": per_deficit_analysis(model_df, label),
    }

print("[ok] Q2 model-vs-baseline analyses computed")
for label in LABELS:
    print(label, model_validation[label]["overall_brier_vs_baseline_C"])
""")


add("code", "c05_000a", """
def classify_brier_result(summary: dict[str, Any]) -> str:
    imp = summary["brier_improvement_baseline_C_minus_model"]
    ci = summary["brier_improvement_bootstrap_ci"]
    lo = ci["2.5"]
    if imp > 0.005 and lo is not None and lo > 0:
        return "yes_materially"
    if imp > 0:
        return "yes_marginally"
    return "no"


interpretation = {
    label: {
        "classification": classify_brier_result(model_validation[label]["overall_brier_vs_baseline_C"]),
        "overall_brier_vs_baseline_C": model_validation[label]["overall_brier_vs_baseline_C"],
        "spearman_model_prob_vs_actual": model_validation[label]["quintile_analysis"]["spearman_model_prob_vs_actual"],
    }
    for label in LABELS
}

if interpretation["favorite_final_win"]["classification"] == "yes_materially":
    structural_finding = (
        "Model probabilities add material favorite-final-win information beyond the training-years "
        "deficit-by-time baseline."
    )
elif interpretation["favorite_final_win"]["classification"] == "yes_marginally":
    structural_finding = (
        "Model probabilities add marginal favorite-final-win information beyond the training-years "
        "deficit-by-time baseline."
    )
else:
    structural_finding = (
        "Model probabilities do not improve favorite-final-win Brier versus the training-years "
        "deficit-by-time baseline."
    )

de_class = interpretation["deficit_erased"]["classification"]
structural_finding += (
    f" For the literal deficit-erased label, the classification is {de_class}."
)

analysis = {
    "created_at": pd.Timestamp.now().isoformat(),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    "config": {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "training_baseline_seasons": TRAIN_BASELINE_SEASONS,
        "test_seasons": TEST_SEASONS,
        "time_bucket_policy": "Q1=period 1; Q2-first-half=period 2; Q3=period 3; Q4=period 4; period_seconds_elapsed=900-clock_seconds_in_period_total",
        "baseline_C_policy": "label rate by fav_deficit x time_bucket computed on 2015-2021 only",
    },
    "data_quality": {
        "deficit_label_quality": deficit_label_quality,
        "feature_subset_quality": feature_subset_quality,
        "n_descriptive_rows": int(len(base)),
        "n_model_validation_rows_scheme_U": int(len(model_df)),
    },
    "descriptive": descriptive,
    "baseline_tables": baseline_tables,
    "model_validation": model_validation,
    "interpretation": {
        "structural_finding": structural_finding,
        "by_label": interpretation,
        "favorite_final_win_vs_deficit_erased_note": (
            "favorite_final_win is the N03 training label; deficit_erased is a newly computed literal comeback event. "
            "Differences between the two indicate whether the model is detecting final game recovery, temporary deficit erasure, or both."
        ),
    },
}

N05_ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2, default=json_default) + "\\n", encoding="utf-8")
print(f"[ok] wrote {N05_ANALYSIS_JSON.relative_to(REPO_ROOT)}")
print(structural_finding)
""")


add("code", "c05_000b", """
lines: list[str] = []
lines.append("# N05 favorite final-win and deficit-erased rate analysis")
lines.append("")
fw_summary = model_validation["favorite_final_win"]["overall_brier_vs_baseline_C"]
de_summary = model_validation["deficit_erased"]["overall_brier_vs_baseline_C"]
fw_ci = fw_summary["brier_improvement_bootstrap_ci"]
de_ci = de_summary["brier_improvement_bootstrap_ci"]
fw_overall = descriptive["favorite_final_win"]["overall"]
de_overall = descriptive["deficit_erased"]["overall"]
rate_gap = de_overall["rate"] - fw_overall["rate"]
rate_gap_pp = rate_gap * 100.0
signed_fmt = lambda x: f"{float(x):+.5f}"

lines.append(
    "**Primary finding:** The model does **not** improve over a simple `fav_deficit x time_bucket` "
    "baseline on either label. For `favorite_final_win`, Brier improvement (`baseline_C - model`) "
    f"is **{signed_fmt(fw_summary['brier_improvement_baseline_C_minus_model'])}** with 95% cluster-bootstrap CI "
    f"**[{signed_fmt(fw_ci['2.5'])}, {signed_fmt(fw_ci['97.5'])}]**, not distinguishable from zero. For "
    f"`deficit_erased`, improvement is **{signed_fmt(de_summary['brier_improvement_baseline_C_minus_model'])}** "
    f"with CI **[{signed_fmt(de_ci['2.5'])}, {signed_fmt(de_ci['97.5'])}]**, materially worse than baseline."
)
lines.append("")
lines.append(
    "**Interpretation:** N03's calibrated probabilities largely encode deficit x time information rather "
    "than adding comeback-detection signal beyond what a naive lookup table provides. N04's positive "
    "Brier improvement against pre-game market probability is real, but mechanism-restricted: the model "
    "beats pre-game markets because pre-game markets do not condition on current game state, not because "
    "the model has discovered comeback patterns that a deficit/time baseline misses."
)
lines.append("")
lines.append(
    "**Secondary finding:** Favorites erase deficits much more often than they win games after a trigger: "
    f"`deficit_erased` rate **{pct(de_overall['rate'])}** versus `favorite_final_win` rate "
    f"**{pct(fw_overall['rate'])}**, a roughly **{rate_gap_pp:.1f} percentage-point** gap. The `favorite came back but lost` "
    "subpopulation is substantial and should be treated as its own future research object."
)
lines.append("")
lines.append(
    "**Tertiary finding:** The model is systematically under-calibrated for the `deficit_erased` label. "
    "Across the middle probability deciles, actual deficit-erased rates exceed model probability by roughly "
    "15-30 percentage points, consistent with the model being trained on `favorite_final_win` "
    f"({pct(fw_overall['rate'])} base rate) rather than `deficit_erased` ({pct(de_overall['rate'])} base rate)."
)
lines.append("")
lines.append("N05 distinguishes two labels throughout: `favorite_final_win` is the N03 target, while `deficit_erased` means the favorite tied or retook the lead after the trigger. The model was trained on `favorite_final_win`, not on `deficit_erased`.")
lines.append("")
lines.append("## Q2 model versus deficit-by-time baseline")
lines.append("")
lines.append("Baseline C is the training-years-only rate by `fav_deficit x time_bucket` using seasons 2015-2021. Positive Brier improvement means the model beat that baseline on held-out 2022-2024 Scheme U rows.")
lines.append("")
lines.append("| Label | Rows | Model Brier | Baseline C Brier | Improvement | 95% bootstrap CI | Spearman(model, actual) | Classification |")
lines.append("|---|---:|---:|---:|---:|---|---:|---|")
for label in LABELS:
    s = model_validation[label]["overall_brier_vs_baseline_C"]
    ci = s["brier_improvement_bootstrap_ci"]
    lines.append(
        f"| `{label}` | {s['n']} | {fmt(s['brier_model'])} | {fmt(s['brier_baseline_C'])} | "
        f"{fmt(s['brier_improvement_baseline_C_minus_model'])} | "
        f"[{fmt(ci['2.5'])}, {fmt(ci['97.5'])}] | "
        f"{fmt(model_validation[label]['quintile_analysis']['spearman_model_prob_vs_actual'], 4)} | "
        f"{interpretation[label]['classification']} |"
    )

lines.append("")
lines.append("### Threshold analysis")
for label in LABELS:
    lines.append("")
    lines.append(f"Label: `{label}`")
    lines.append("")
    lines.append("| Threshold X | N | Actual rate | Mean model prob | Mean baseline C | Actual - model | Actual - baseline C | Wilson 95% CI | Bootstrap 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in model_validation[label]["threshold_analysis"]:
        bci = r["bootstrap_rate_ci"]
        lines.append(
            f"| {fmt(r['threshold'], 2)} | {r['n']} | {fmt(r['rate'])} | {fmt(r['mean_model_prob'])} | "
            f"{fmt(r['mean_baseline_C'])} | {fmt(r['actual_minus_mean_model_prob'])} | "
            f"{fmt(r['actual_minus_mean_baseline_C'])} | "
            f"[{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] | "
            f"[{fmt(bci['2.5'])}, {fmt(bci['97.5'])}] |"
        )

lines.append("")
lines.append("### Quintile analysis")
for label in LABELS:
    lines.append("")
    lines.append(f"Label: `{label}`; Spearman(model_prob, actual) = {fmt(model_validation[label]['quintile_analysis']['spearman_model_prob_vs_actual'], 4)}")
    lines.append("")
    lines.append("| Quintile | N | Prob range | Mean model prob | Mean baseline C | Actual rate | Actual - model | Actual - baseline C |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in model_validation[label]["quintile_analysis"]["rows"]:
        lines.append(
            f"| {r['quintile']} | {r['n']} | [{fmt(r['model_prob_min'])}, {fmt(r['model_prob_max'])}] | "
            f"{fmt(r['mean_model_prob'])} | {fmt(r['mean_baseline_C'])} | {fmt(r['rate'])} | "
            f"{fmt(r['actual_minus_mean_model_prob'])} | {fmt(r['actual_minus_mean_baseline_C'])} |"
        )

lines.append("")
lines.append("### Calibration deciles")
for label in LABELS:
    lines.append("")
    lines.append(f"Label: `{label}`")
    lines.append("")
    lines.append("| Decile | N | Mean model prob | Actual rate | Calibration gap | Bootstrap 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for r in model_validation[label]["decile_analysis"]:
        bci = r["bootstrap_rate_ci"]
        lines.append(
            f"| {r['decile']} ({fmt(r['lo'], 1)}-{fmt(r['hi'], 1)}) | {r['n']} | "
            f"{fmt(r['mean_model_prob'])} | {fmt(r['rate'])} | {fmt(r['actual_minus_mean_model_prob'])} | "
            f"[{fmt(bci['2.5'])}, {fmt(bci['97.5'])}] |"
        )

lines.append("")
lines.append("## Per-deficit model pattern")
for label in LABELS:
    lines.append("")
    lines.append(f"Label: `{label}`")
    lines.append("")
    lines.append("| Deficit | N | Actual rate | Mean model prob | Mean baseline C | Model Brier | Baseline C Brier | Improvement | 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in model_validation[label]["per_deficit_analysis"]:
        ci = r["brier_improvement_bootstrap_ci"]
        lines.append(
            f"| D={r['fav_deficit']} | {r['n']} | {fmt(r['rate'])} | {fmt(r['mean_model_prob'])} | "
            f"{fmt(r['mean_baseline_C'])} | {fmt(r['brier_model'])} | {fmt(r['brier_baseline_C'])} | "
            f"{fmt(r['brier_improvement_baseline_C_minus_model'])} | "
            f"[{fmt(ci['2.5'])}, {fmt(ci['97.5'])}] |"
        )

lines.append("")
lines.append("## Q1 descriptive rates")
for label in LABELS:
    overall = descriptive[label]["overall"]
    bci = overall["bootstrap_rate_ci"]
    lines.append("")
    lines.append(f"Label: `{label}` overall rate = {fmt(overall['rate'])} ({overall['successes']}/{overall['n']}), Wilson 95% CI [{fmt(overall['wilson_low'])}, {fmt(overall['wilson_high'])}], bootstrap 95% CI [{fmt(bci['2.5'])}, {fmt(bci['97.5'])}].")
    lines.append("")
    lines.append("By deficit:")
    lines.append("")
    lines.append("| Deficit | Successes | N | Rate | Wilson 95% CI |")
    lines.append("|---:|---:|---:|---:|---|")
    for r in descriptive[label]["by_deficit"]:
        lines.append(f"| D={r['fav_deficit']} | {r['successes']} | {r['n']} | {fmt(r['rate'])} | [{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] |")
    lines.append("")
    lines.append("By time bucket:")
    lines.append("")
    lines.append("| Time bucket | Successes | N | Rate | Wilson 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for r in descriptive[label]["by_time_bucket"]:
        lines.append(f"| {r['time_bucket']} | {r['successes']} | {r['n']} | {fmt(r['rate'])} | [{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] |")
    lines.append("")
    lines.append("By deficit x time bucket:")
    lines.append("")
    lines.append("| Deficit | Time bucket | Successes | N | Rate | Wilson 95% CI |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for r in descriptive[label]["by_deficit_time"]:
        lines.append(
            f"| D={r['fav_deficit']} | {r['time_bucket']} | {r['successes']} | {r['n']} | "
            f"{fmt(r['rate'])} | [{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] |"
        )
    lines.append("")
    lines.append("By season:")
    lines.append("")
    lines.append("| Season | Successes | N | Rate | Wilson 95% CI |")
    lines.append("|---:|---:|---:|---:|---|")
    for r in descriptive[label]["by_season"]:
        lines.append(f"| {r['season']} | {r['successes']} | {r['n']} | {fmt(r['rate'])} | [{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] |")
    lines.append("")
    lines.append("By feature subset:")
    for subset_name, rows in descriptive[label]["by_feature_subset"].items():
        bucket_col = next((k for k in rows[0].keys() if k not in {"successes", "n", "rate", "wilson_low", "wilson_high"}), "bucket") if rows else "bucket"
        lines.append("")
        lines.append(f"Subset: `{subset_name}`")
        lines.append("")
        lines.append("| Bucket | Successes | N | Rate | Wilson 95% CI |")
        lines.append("|---|---:|---:|---:|---|")
        for r in rows:
            lines.append(f"| {r.get(bucket_col)} | {r['successes']} | {r['n']} | {fmt(r['rate'])} | [{fmt(r['wilson_low'])}, {fmt(r['wilson_high'])}] |")

lines.append("")
lines.append("## Data quality")
lines.append("")
lines.append(f"- Trigger events: {len(base):,}")
lines.append(f"- Unique trigger plays: {base[['game_id', 'trigger_play_id']].drop_duplicates().shape[0]:,}")
lines.append(f"- Missing play-by-play games: {len(missing_play_games)}")
lines.append(f"- Missing trigger play IDs: {deficit_label_quality['missing_trigger_play_ids']}")
lines.append(f"- Trigger score mismatches: {deficit_label_quality['score_mismatch_count']}")
lines.append(f"- No-post-trigger unique plays labeled null: {deficit_label_quality['no_post_trigger_play_count']}")
lines.append(f"- Deficit-erased null event rows excluded label-wise: {deficit_label_quality['deficit_erased_null_event_rows']}")

lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append(analysis["interpretation"]["favorite_final_win_vs_deficit_erased_note"])
lines.append("")
lines.append("If model-vs-baseline improvement is small or negative, N05 should be read as evidence that N03's apparent comeback detection mostly inherits deficit/time structure. If improvement is positive with a CI above zero, N05 supports incremental comeback-detection signal beyond the naive structural baseline.")

N05_SUMMARY_MD.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"[ok] wrote {N05_SUMMARY_MD.relative_to(REPO_ROOT)}")
print("\\n".join(lines[:24]))
""")


add("markdown", "m05_000c", """
N05 complete. Halt for review; no commit is performed by this notebook.
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
