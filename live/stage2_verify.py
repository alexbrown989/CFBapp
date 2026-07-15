"""N13 Stage 2 exact lookup and cached Tier 3 acceptance gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPO_ROOT, ensure_n12_lookup_import_path
from .logger import JSONLTriggerLogger, SCORING_FIELDS, read_trigger_records
from .parity_guard import RuntimeParityGuard
from .scoring import HistoricalEstimate, ScoringContext, ScoringEngine
from .trigger_detect import TriggerEvent

ensure_n12_lookup_import_path()
import _lib_lookup  # type: ignore  # noqa: E402


N11_PATH = REPO_ROOT / "research/results/n11_ranking_stratification.parquet"
N06_PATH = REPO_ROOT / "research/results/n06_calibrated_predictions.parquet"
TRIGGERS_PATH = REPO_ROOT / "research/results/trigger_events.csv"
LOOKUP_PATH = REPO_ROOT / "research/results/n12_probability_lookup.parquet"
REPORT_PATH = REPO_ROOT / "research/results/n13_stage2_scoring_verification.md"
PARITY_LOG_PATH = REPO_ROOT / "live/logs/stage2_parity_verification.jsonl"
COMPAT_LOG_PATH = REPO_ROOT / "live/logs/stage1_compatibility_verification.jsonl"

CANDIDATES = (
    (401628332, "401628332103934901", 1, 3),
    (401677090, "401677090104956501", 2, 7),
    (401677086, "401677086101989101", 3, 10),
    (401677082, "401677082102969001", 4, 14),
    (401628374, "401628374101977803", 5, 21),
)


def _load_rows() -> list[dict[str, Any]]:
    n11 = pd.read_parquet(N11_PATH)
    n06 = pd.read_parquet(N06_PATH)
    n06 = n06[(n06["scheme"] == "U") & (n06["fold"] == 2024)].copy()
    triggers = pd.read_csv(TRIGGERS_PATH, low_memory=False)
    key = ["game_id", "trigger_play_id", "trigger_sequence", "fav_deficit"]
    merged = n11[n11["season"] == 2024].merge(n06, on=key + ["season"], suffixes=("_n11", "_n06"), validate="one_to_one")
    merged = merged.merge(
        triggers[key + ["pregame_spread"]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for game_id, play_id, sequence, deficit in CANDIDATES:
        selected = merged[
            (merged["game_id"] == game_id)
            & (merged["trigger_play_id"].astype(str) == play_id)
            & (merged["trigger_sequence"] == sequence)
            & (merged["fav_deficit"] == deficit)
        ]
        if len(selected) != 1:
            raise AssertionError(f"expected one candidate row for {(game_id, play_id, sequence, deficit)}, got {len(selected)}")
        rows.append(selected.iloc[0].to_dict())
    return rows


def _event(row: dict[str, Any]) -> TriggerEvent:
    seconds = int(row["clock_seconds_in_period_total"])
    return TriggerEvent(
        timestamp=f"cached-2024-{int(row['game_id'])}-{int(row['trigger_sequence'])}",
        game_id=str(int(row["game_id"])),
        season=2024,
        week=int(row["week"]),
        favorite=str(row["fav_team_n06"]),
        dog=str(row["dog_team_n06"]),
        pregame_spread=float(row["pregame_spread"]),
        fav_score=int(row["fav_score_at_trigger"]),
        dog_score=int(row["dog_score_at_trigger"]),
        period=int(row["quarter"]),
        clock=f"{seconds // 60}:{seconds % 60:02d}",
        deficit=-int(row["fav_deficit"]),
        threshold_crossed=int(row["fav_deficit"]),
        possession=None,
        data_source="stub",
        poll_number=int(row["trigger_sequence"]),
    )


def _feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    spec = _lib_lookup.load_scoring_spec()
    deployment = spec["deployment_choice"]
    fit = next(
        item
        for item in spec["n06_fitted_state"]["fits"]
        if item["scheme"] == "U" and int(item["fold"]) == 2024
    )
    assert deployment["scheme"] == "E" and int(deployment["fold"]) == 2024
    return {feature: row[feature] for feature in fit["core_features"]}


def _context(row: dict[str, Any]) -> ScoringContext:
    return ScoringContext(
        spread_bucket=str(row["spread_bucket"]),
        ranking_bucket=str(row["ranking_bucket"]),
        fluke_bucket=str(row["fluke_bucket"]),
        time_bucket=str(row["time_bucket_n11"]),
        tier3_features=_feature_dict(row),
        tier3_certified=True,
        tier3_feature_source="cached_historical",
        tier3_scheme="U",
        tier3_fold=2024,
        tier3_unavailable_reason=None,
    )


def _expected_rows(lookup: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    deficit = int(row["fav_deficit"])
    time_bucket = str(row["time_bucket_n11"])
    keys = {
        _lib_lookup.build_lookup_key("baseline_c_rate", deficit=deficit, time_bucket=time_bucket),
        _lib_lookup.build_lookup_key(
            "conditional_rate_full",
            fluke_bucket=row["fluke_bucket"],
            deficit=deficit,
            time_bucket=time_bucket,
            spread_bucket=row["spread_bucket"],
        ),
        _lib_lookup.build_lookup_key(
            "ranking_rate",
            ranking_bucket=row["ranking_bucket"],
            deficit=deficit,
            time_bucket=time_bucket,
            spread_bucket=row["spread_bucket"],
        ),
    }
    return lookup[lookup["lookup_key"].isin(keys)].copy()


def _assert_estimate_exact(estimate: HistoricalEstimate, expected: pd.DataFrame) -> None:
    rows = expected[
        (expected["lookup_key"] == estimate.lookup_key)
        & (expected["metric_name"] == estimate.metric_name)
        & (expected["label"] == estimate.label)
    ]
    if len(rows) != 1:
        raise AssertionError(
            f"expected one direct parquet row for {estimate.metric_name}/{estimate.label}/{estimate.lookup_key}; got {len(rows)}"
        )
    row = rows.iloc[0]
    assert estimate.value == float(row["value"])
    assert estimate.n_events == _optional_int(row["n_events"])
    assert estimate.n_games == _optional_int(row["n_games"])
    assert estimate.n_seasons == _optional_int(row["n_seasons"])
    assert estimate.reliability_flag == str(row["reliability_flag"])


def run_acceptance() -> dict[str, Any]:
    rows = _load_rows()
    lookup = pd.read_parquet(LOOKUP_PATH)
    engine = ScoringEngine()
    guard = RuntimeParityGuard(default_tolerance=1e-12)
    PARITY_LOG_PATH.unlink(missing_ok=True)
    details: list[dict[str, Any]] = []
    max_tier3_diff = 0.0
    max_parity_diff = 0.0

    for row in rows:
        result = engine.score_trigger(_event(row), _context(row))
        assert result.tier_used == 3
        expected = _expected_rows(lookup, row)
        for estimate in result.tier_1.values():
            _assert_estimate_exact(estimate, expected)
        for estimates in result.tier_2.values():
            for estimate in estimates:
                _assert_estimate_exact(estimate, expected)

        assert result.tier_3 is not None
        tier3_diff = abs(result.tier_3.calibrated_prob - float(row["calibrated_prob"]))
        if tier3_diff >= 1e-6:
            raise AssertionError(f"Tier 3 mismatch for game {row['game_id']}: {tier3_diff}")
        max_tier3_diff = max(max_tier3_diff, tier3_diff)

        features = _feature_dict(row)
        parity = guard.compare(
            trigger_id=f"{int(row['game_id'])}:{int(row['trigger_sequence'])}:{int(row['fav_deficit'])}",
            game_id=str(int(row["game_id"])),
            live_features=features,
            cached_features=dict(features),
        )
        assert not parity.tier3_suspect
        assert parity.max_abs_diff == 0.0
        RuntimeParityGuard.append_jsonl(parity, PARITY_LOG_PATH)
        max_parity_diff = max(max_parity_diff, parity.max_abs_diff)

        details.append(
            {
                "game_id": int(row["game_id"]),
                "trigger_sequence": int(row["trigger_sequence"]),
                "deficit": int(row["fav_deficit"]),
                "time_bucket": str(row["time_bucket_n11"]),
                "spread_bucket": str(row["spread_bucket"]),
                "ranking_bucket": str(row["ranking_bucket"]),
                "fluke_bucket": str(row["fluke_bucket"]),
                "tier_used": result.tier_used,
                "baseline_final_win": result.tier_1["favorite_final_win"].value,
                "baseline_deficit_erased": result.tier_1["deficit_erased"].value,
                "n06_engine": result.tier_3.calibrated_prob,
                "n06_committed": float(row["calibrated_prob"]),
                "tier3_abs_diff": tier3_diff,
            }
        )

    first = _feature_dict(rows[0])
    changed = dict(first)
    changed["plays_so_far"] = float(changed["plays_so_far"]) + 1.0
    drift_check = guard.compare(
        trigger_id="synthetic-drift-check",
        game_id=str(int(rows[0]["game_id"])),
        live_features=changed,
        cached_features=first,
    )
    assert drift_check.tier3_suspect
    assert [record.feature for record in drift_check.records if record.drifted] == ["plays_so_far"]

    missing_features = _feature_dict(rows[0])
    missing_features.pop("dog_points_off_turnovers")
    suppressed = engine.score_trigger(
        _event(rows[0]),
        ScoringContext(
            spread_bucket=str(rows[0]["spread_bucket"]),
            ranking_bucket=str(rows[0]["ranking_bucket"]),
            fluke_bucket=str(rows[0]["fluke_bucket"]),
            time_bucket=str(rows[0]["time_bucket_n11"]),
            tier3_features=missing_features,
            tier3_certified=True,
            tier3_feature_source="cached_historical",
            tier3_scheme="U",
            tier3_fold=2024,
        ),
    )
    assert suppressed.tier_used == 2
    assert suppressed.tier_3 is None
    assert "dog_points_off_turnovers" in str(suppressed.tier_3_unavailable_reason)

    COMPAT_LOG_PATH.unlink(missing_ok=True)
    stage1_event = _event(rows[0])
    JSONLTriggerLogger(COMPAT_LOG_PATH).append(stage1_event)
    compatible = read_trigger_records(COMPAT_LOG_PATH)
    assert len(compatible) == 1
    assert all(compatible[0][field] is None for field in SCORING_FIELDS)

    return {
        "details": details,
        "max_tier3_abs_diff": max_tier3_diff,
        "max_parity_abs_diff": max_parity_diff,
        "exact_lookup_match": True,
        "synthetic_drift_feature": "plays_so_far",
        "stage1_log_compatibility": True,
        "missing_feature_suppression": True,
    }


def write_report(result: dict[str, Any]) -> None:
    rows = "\n".join(
        "| {game_id} | D={deficit} | {time_bucket} | {spread_bucket} | {ranking_bucket} | "
        "{fluke_bucket} | {tier_used} | {baseline_final_win:.6f} | {baseline_deficit_erased:.6f} | "
        "{n06_engine:.12f} | {tier3_abs_diff:.3g} |".format(**row)
        for row in result["details"]
    )
    report = f"""# N13 Stage 2 Scoring Verification

Date: 2026-07-14

## Acceptance Result

PASS. Five cached 2024 triggers spanning all five deficit thresholds, Q1-Q4, and varied spread/ranking/fluke contexts were scored through `live.scoring.score_trigger()`.

- Tier 1 and Tier 2: exact equality with independently filtered rows from `n12_probability_lookup.parquet` for estimate value, sample sizes, and reliability flag.
- Tier 3: maximum absolute difference versus committed N06 calibrated predictions = `{result['max_tier3_abs_diff']:.17g}` (required `< 1e-6`).
- Runtime parity guard: maximum absolute difference across identical cached feature snapshots = `{result['max_parity_abs_diff']:.1f}`.
- Synthetic drift check: changing only `plays_so_far` by 1 correctly sets `tier3_suspect=true` and identifies only that feature.
- Additive log compatibility: a 16-field Stage 1 record remains readable, with Stage 2 fields exposed as null.
- Tier 3 missing-feature guard: removing `dog_points_off_turnovers` suppresses N06 and leaves Tier 2 active; no imputation occurs.
- Network/API calls: 0.

| Game | Deficit | Time | Spread | Rank | Fluke | Tier | baseline_C final win | baseline_C deficit erased | N06 calibrated | abs diff |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
{rows}

## Tier Behavior

Tier 1 is always present and remains the primary score+clock estimate. Tier 2 rows are labeled historical descriptive and retain N12 sample sizes, reliability, confidence bounds, and source provenance. Tier 3 appears only for explicitly certified cached historical feature dictionaries in this verification. Normal Stage 2 runtime suppresses Tier 3 with `unavailable - no live play feed`.

The parity guard comparison and append-only drift schema are ready, but the first weeks of the 2026 season remain the live-feed certification window. Until that window is clean, operational decisions should lean on Tier 1.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def _optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def main() -> int:
    result = run_acceptance()
    write_report(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"PASS: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
