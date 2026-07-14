# N13 Stage 1 Replay Verification

Date: 2026-07-14

## Result

PASS. The source-agnostic Stage 1 detector replayed cached 2024 game `401628374` (No. 2 Georgia at Alabama), built its watch-list entry from cached games, lines, and the week-5 AP Top 25 poll, and emitted the expected trigger sequence.

CFBD applies some completed-drive score values to earlier plays in a drive. The replay therefore advances the scoreboard only on records marked `scoring=true`, matching actual scoring state transitions rather than treating those backward-stamped values as live score changes.

| Threshold | Observed state | Favorite-dog score | Poll |
|---:|---|---:|---:|
| 3 | Q1 10:11 | 0-7 | 1 |
| 7 | Q1 10:11 | 0-7 | 1 |
| 10 | Q1 4:39 | 0-14 | 2 |
| 14 | Q1 4:39 | 0-14 | 2 |
| 21 | Q1 2:21 | 0-21 | 3 |
| 3 | Q4 2:18 | 34-41 | 12 |
| 7 | Q4 2:18 | 34-41 | 12 |

- Scoring states replayed: 12
- Initial-descent thresholds emitted once: 3, 7, 10, 14, 21
- Multi-threshold crossings: 0-7 emitted D=3 and D=7; 0-14 emitted D=10 and D=14
- Real-game re-fire after Georgia recovered to a 34-33 lead: D=3 and D=7 at Q4 2:18
- Synthetic re-fire test after recovery: [3, 7, 3, 7]
- JSONL records: 7
- JSONL schema: PASS, all 16 required fields present
- Data source recorded as `stub`
- Network/API calls: 0

## Acceptance

Trigger timing, one-fire deduplication, recovery-based re-arming, multi-threshold crossing, watch-list construction, and append-only local log schema all pass. `ScoreboardLive` was not activated.
