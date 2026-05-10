# trigger_outcomes.csv — schema sidecar

**Generated:** 2026-05-09 16:16:27 Pacific Daylight Time
**Source notebook:** `research/notebooks/01_trigger_events.ipynb`
**Source commit:** `9df25b9e84b54d9d83259665b0294d4852bc35c4`
**Trigger table version:** `v1`

This file documents the V5 **OUTCOME / LABEL** block that was **split out** of
`trigger_events.csv` for R3 structural safety. Feature notebooks load
`trigger_events.csv` **only**. Notebook 03 (and later) loads this file and joins
on `(game_id, fav_deficit)` after computing features — never merge labels before
feature extraction.

## Join contract

```python
df = trigger_events.merge(
    trigger_outcomes,
    on=["game_id", "fav_deficit"],
    how="inner",
    validate="one_to_one",
)
```

Row count equals `trigger_events.csv` row count (one outcome row per trigger row).

## Column dictionary

| Column | Type | Unit | Range / sign | Category | Description |
|---|---|---|---|---|---|
| `game_id` | INTEGER | — | CFBD game ID | IDENTIFIER / JOIN | Natural key with `fav_deficit`. Must match `trigger_events.game_id`. |
| `fav_deficit` | INTEGER | points | {3, 7, 10, 14, 21} | IDENTIFIER / JOIN | Natural key with `game_id`. Must match `trigger_events.fav_deficit`. |
| `final_fav_won` | BOOLEAN | — | True | False | NULL | LABEL | Game-final outcome: True iff favorite's final score > underdog's after full game (including OT). R3-FORBIDDEN as a model **feature** — use only as prediction **target** in N03+ after an explicit join. |
| `final_fav_score` | INTEGER | points | >=0 | NULL | LABEL | Favorite's final score. R3-FORBIDDEN as model feature. |
| `final_dog_score` | INTEGER | points | >=0 | NULL | LABEL | Underdog's final score. R3-FORBIDDEN as model feature. |
| `margin_of_defeat` | INTEGER | points | >0 | NULL | LABEL | If favorite lost: dog_final - fav_final (positive). NULL if favorite won or tie/unknown. R3-FORBIDDEN as model feature. |


## Generation provenance

- Commit hash: `9df25b9e84b54d9d83259665b0294d4852bc35c4`
- Outcome rows: 11,416
