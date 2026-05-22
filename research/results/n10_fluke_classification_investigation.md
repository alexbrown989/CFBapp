# N10 fluke classification investigation

Date: 2026-05-21

## Scope

This document manually inspects 20 trigger games after the N10 fluke-classification sanity gate failed. The inspection uses the current Option C N10 classification path: N07 percentage components when present, `attribution_unclear` for missing components, and the five known `dog_points_from_explosives_pct > 1` rows treated as fluky because explosive points exceeded visible dog score.

The sample is deterministic: one trigger per unique game, `random_state=42`, drawn separately from `fluky_lead` rows with `fluke_composite >= 0.60` and `sustained_lead` rows with `fluke_composite < 0.30`.

## Executive finding

The current N07-component classifier is directionally useful but not clean enough to support a strong semantic claim that `fluky_lead` means visibly cheap scoring in every case. In the 20-game manual sample, most `fluky_lead` examples contain meaningful explosive/return/turnover-created scoring, but several are mixed with ordinary field goals or sustained drives. The `sustained_lead` examples are mostly sustained, but some include short-field or explosive elements that the N07 components do not fully surface.

The yards-per-point sanity check does not separate the buckets cleanly because N07 classifies point-attribution buckets, not total drive efficiency. A fluky lead can include one explosive touchdown plus enough ordinary offense or field goals to raise total yards per point. Conversely, a sustained lead can be built by short fields and field goals with low total yards per point while still avoiding the N07 explosive/return thresholds.

Recommendation: **Option C, but with an additional direct-yardage guard for headline claims.** Keep the existing N07-based components as descriptive dashboard columns, but do not use the current `fluky_lead` bucket alone as the headline hypothesis cell. For N10 headline testing, define a stricter `clear_fluky_lead` subgroup that requires both component evidence (`fluke_composite >= 0.60`) and a cheap-yardage signature (for example, dog completed-drive yards per point below the sustained-lead median, or manual/drive-derived yards-per-point below a locked threshold). This avoids abandoning the pre-registered fluke components while preventing the headline test from resting on a semantically noisy bucket.

Alternatives: Option A (keep as-is) is acceptable only if the report explicitly says the bucket is "N07-attributed fluke share" rather than football-obvious cheap scoring. Option B (redefine solely using yards per point) is too blunt because it would miss defensive/return and one-play explosive mechanisms that can still accumulate yardage elsewhere. A component modification pass is worthwhile later, but it would be a new feature-definition decision rather than an N10 patch.

## Full-bucket aggregate cross-check

| bucket | n | mean_true_scrimmage_ypp | median_true_scrimmage_ypp | mean_drive_ypp | median_drive_ypp | mean_dog_drives | mean_manual_cheap_point_share | median_manual_cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fluky_lead | 3318 | 11.638 | 10.961 | 9.117 | 8.904 | 5.489 | 0.346 | 0.333 |
| sustained_lead | 2194 | 11.934 | 10.714 | 5.449 | 4.297 | 2.196 | 0.052 | 0.000 |

Notes: true scrimmage yards exclude kicks, punts, field-goal distance, penalties, and administrative plays. Completed-drive yards come from CFBD drive summaries before the trigger drive. The failed N10 gate used a broader play-level yard sum; this investigation uses football-yardage definitions for manual interpretation.

## Sample-level summary

| group | yes | mixed | no | mean_scr_ypp | mean_drive_ypp | mean_cheap_share |
| --- | --- | --- | --- | --- | --- | --- |
| fluky_lead | 6 | 3 | 1 | 10.668 | 8.290 | 0.641 |
| sustained_lead | 10 | 0 | 0 | 10.248 | 5.631 | 0.000 |

`yes/mixed/no` means whether the manual scoring ledger matches the expected bucket semantics for that sample group.

## fluky_lead inspections

### fluky_lead #1: 2016 Virginia vs Duke (game_id 400869454)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q2 clock_seconds=241 | 13 | 1.000 | 0.538 | 0.000 | 0.462 | 203.000 | 31 | 15.615 | 108.000 | 6 | 8.308 | 1.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 00:38 | 8 | 6 | Passing Touchdown | 28 | False | True | explosive offense | David Eldridge 28 Yd pass from Kurt Benkert (Two-Point Run Conversion Failed) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #2: 2016 Temple vs Navy (game_id 400926947)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=21, seq=5, Q2 clock_seconds=824 | 21 | 0.905 | 0.333 | 0.000 | 0.571 | 148.000 | 22 | 7.048 | 134.000 | 2 | 6.381 | 1.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 09:43 | 1 | 7 | Rushing Touchdown | 15 | False | True | explosive offense | Jahad Thomas run for 15 yds for a TD, (Aaron Boumerhi KICK) |
| Q1 03:01 | 3 | 7 | Passing Touchdown | 22 | False | True | explosive offense | Phillip Walker pass complete to Ventell Bryant for 22 yds for a TD, (Aaron Boumerhi KICK) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #3: 2017 Rutgers vs Illinois (game_id 400935378)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=14, seq=4, Q3 clock_seconds=251 | 28 | 0.643 | 0.000 | 0.000 | 0.643 | 276.000 | 39 | 9.857 | 254.000 | 9 | 9.071 | 0.333 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 07:21 | 5 | 7 | Rushing Touchdown | 19 | False | True | explosive offense | Raheem Blackshear run for 19 yds for a TD, (Andrew Harte KICK) |
| Q2 13:07 | 7 | 7 | Rushing Touchdown | 5 | False | False | sustained offense | Josh Hicks run for 5 yds for a TD, (Andrew Harte KICK) |
| Q2 04:59 | 9 | 7 | Rushing Touchdown | 7 | False | False | sustained offense | Gus Edwards run for 7 yds for a TD, (Andrew Harte KICK) |

Manual judgment: **MIXED: a cheap component exists, but sustained/FG scoring is material.**

### fluky_lead #4: 2017 Marshall vs Colorado State (game_id 400953324)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=10, seq=3, Q3 clock_seconds=732 | 28 | 0.643 | 0.000 | 0.000 | 0.643 | 285.000 | 30 | 10.179 | 280.000 | 8 | 10.000 | 0.667 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q2 14:29 | 7 | 7 | Passing Touchdown | 76 | False | True | explosive offense | Chase Litton pass complete to Tyre Brady for 76 yds for a TD, (Kaare Vedvik KICK) |
| Q2 06:07 | 9 | 7 | Passing Touchdown | 15 | False | False | sustained offense | Chase Litton pass complete to Ryan Yurachek for 15 yds for a TD, (Kaare Vedvik KICK) |
| Q2 03:55 | 11 | 7 | Rushing Touchdown | 68 | False | True | explosive offense | Keion Davis run for 68 yds for a TD, (Kaare Vedvik KICK) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #5: 2020 Washington State vs Oregon State (game_id 401249397)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=21, seq=5, Q3 clock_seconds=500 | 28 | 0.643 | 0.000 | 0.000 | 0.643 | 282.000 | 44 | 10.071 | 277.000 | 7 | 9.893 | 0.333 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 03:50 | 5 | 7 | Passing Touchdown | 29 | False | True | explosive offense | Jayden de Laura pass complete to Travell Harris for 29 yds for a TD OREGON ST Penalty, Defensive offside ( Yards) dec? |
| Q2 00:37 | 11 | 7 | Rushing Touchdown | 3 | False | False | sustained offense | Deon McIntosh run for 3 yds for a TD, (Blake Mazza KICK) |
| Q3 12:10 | 14 | 7 | Rushing Touchdown | 5 | False | False | sustained offense | Jayden de Laura run for 5 yds for a TD, (Blake Mazza KICK) |

Manual judgment: **MIXED: a cheap component exists, but sustained/FG scoring is material.**

### fluky_lead #6: 2020 Central Michigan vs Western Michigan (game_id 401249872)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=10, seq=3, Q1 clock_seconds=678 | 14 | 0.929 | 0.500 | 0.000 | 0.429 | 97.000 | 7 | 6.929 | 75.000 | 1 | 5.357 | 1.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 13:57 | 1 | 7 | Rushing Touchdown | 65 | False | True | explosive offense | Kalil Pimpleton run for 65 yds for a TD (Marshall Meeder KICK) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #7: 2021 Rice vs Louisiana Tech (game_id 401282271)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q2 clock_seconds=647 | 14 | 0.929 | 0.500 | 0.000 | 0.429 | 148.000 | 15 | 10.571 | 119.000 | 3 | 8.500 | 1.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 01:07 | 5 | 7 | Rushing Touchdown | 70 | False | True | explosive offense | Cameron Montgomery 70 Yd Run (Christian VanSickle Kick) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #8: 2021 Michigan vs Ohio State (game_id 401282781)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=10, seq=3, Q3 clock_seconds=349 | 28 | 0.643 | 0.000 | 0.000 | 0.643 | 400.000 | 44 | 14.286 | 298.000 | 7 | 10.643 | 0.667 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 10:12 | 1 | 7 | Rushing Touchdown | 14 | False | True | explosive offense | A.J. Henning run for 14 yds for a TD (Jake Moody KICK) |
| Q2 03:51 | 9 | 7 | Rushing Touchdown | 1 | False | False | sustained offense | Hassan Haskins run for 1 yd for a TD (Jake Moody KICK) |
| Q3 11:50 | 13 | 7 | Rushing Touchdown | 13 | False | True | explosive offense | Hassan Haskins run for 13 yds for a TD (Jake Moody KICK) |

Manual judgment: **YES: scoring ledger is mostly cheap/explosive/return by points.**

### fluky_lead #9: 2022 Nebraska vs Minnesota (game_id 401405128)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=10, seq=3, Q1 clock_seconds=258 | 10 | 0.600 | 0.000 | 0.000 | 0.600 | 125.000 | 19 | 12.500 | 75.000 | 1 | 7.500 | 0.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 10:46 | 1 | 7 | Rushing Touchdown | 2 | False | False | sustained offense | Chubba Purdy run for 2 yds for a TD, (Timmy Bleekrode KICK) |

Manual judgment: **NO: manual scoring ledger is mostly sustained/FG despite N07 classification.**

### fluky_lead #10: 2023 Pittsburgh vs Louisville (game_id 401525525)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q3 clock_seconds=219 | 24 | 0.792 | 0.292 | 0.000 | 0.500 | 231.000 | 43 | 9.625 | 174.000 | 8 | 7.250 | 0.412 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 02:37 | 7 | 7 | Passing Touchdown | 46 | False | True | explosive offense | Christian Veilleux pass complete to Bub Means for 46 yds for a TD (Ben Sauls KICK) |
| Q2 01:23 | 11 | 7 | Rushing Touchdown | 1 | False | False | sustained offense | C'Bo Flemister run for 1 yd for a TD (Ben Sauls KICK) |
| Q3 11:32 | 15 | 3 | Field Goal Good | 46 | False | False | field goal | Ben Sauls 46 yd FG GOOD |

Manual judgment: **MIXED: a cheap component exists, but sustained/FG scoring is material.**

## sustained_lead inspections

### sustained_lead #1: 2017 App State vs Wake Forest (game_id 400937478)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=169 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 67.000 | 16 | 11.167 | 53.000 | 2 | 8.833 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #2: 2017 Hawai'i vs Fresno State (game_id 400945297)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=172 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 83.000 | 18 | 11.857 | 36.000 | 1 | 5.143 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #3: 2018 Texas vs TCU (game_id 401013075)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=249 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 113.000 | 16 | 16.143 | 83.000 | 2 | 11.857 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #4: 2018 Wake Forest vs Syracuse (game_id 401013166)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=702 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 50.000 | 9 | 7.143 | 8.000 | 1 | 1.143 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #5: 2019 Duke vs Wake Forest (game_id 401112515)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=24 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 11.000 | 10 | 1.571 | 11.000 | 3 | 1.571 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #6: 2020 Colorado vs Stanford (game_id 401249406)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=200 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 43.000 | 11 | 6.143 | 41.000 | 3 | 5.857 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #7: 2021 UL Monroe vs Texas State (game_id 401309635)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=178 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 87.000 | 15 | 14.500 | 28.000 | 1 | 4.667 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #8: 2022 West Virginia vs Oklahoma State (game_id 401404123)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=431 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 25.000 | 9 | 3.571 | 26.000 | 1 | 3.714 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #9: 2024 SMU vs TCU (game_id 401635557)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=7, seq=2, Q1 clock_seconds=422 | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 41.000 | 10 | 4.100 | 41.000 | 1 | 4.100 | 0.000 |

| clock | drive | pts | type | yds | def/return | explosive | bucket | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 09:52 | 1 | 3 | Field Goal Good | 51 | False | False | field goal | Collin Rogers 51 yd FG GOOD |

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

### sustained_lead #10: 2024 Illinois vs South Carolina (game_id 401677103)

| trigger | dog_score | fluke_composite | turnover_pct | return_pct | explosive_pct | scrimmage_yards | scrimmage_plays | scr_ypp | drive_yards | dog_drives | drive_ypp | cheap_point_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D=3, seq=1, Q1 clock_seconds=0 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 184.000 | 29 | 26.286 | 66.000 | 2 | 9.429 | 0.000 |

_No dog scoring plays found before trigger._

Manual judgment: **YES: scoring ledger is mostly sustained/FG, matching sustained_lead.**

## Pattern explanation

1. The N07 percentages are attribution features, not direct labels of how every dog point was earned. `dog_points_from_explosives_pct` attributes an entire dog scoring drive to explosives if the drive contains an explosive play; that can include otherwise normal drives. `dog_points_from_turnovers_pct` and returns capture different mechanisms and may overlap conceptually with short fields rather than visible scoring-play yardage.
2. Yards per point is sensitive to denominator timing. A dog can have a fluky 7-point return or explosive TD and still add a field goal or ordinary touchdown before the trigger, raising total yards per point. A dog can also have a sustained-looking lead from short fields or field goals with low yards per point.
3. The manual `cheap_point_share` separates the sample more cleanly than yards per point, but it requires play-by-play scoring attribution and is not currently a committed N07 feature. Using it as a headline classifier would be a new definition, not a trivial bug fix.

## Recommendation

Choose **Option C with a stricter headline guard**: retain `attribution_unclear`, keep the N07 component classifier for descriptive tables, and add a pre-declared `clear_fluky_lead` headline subset that requires both N07 component evidence and low drive/scrimmage yards per point. If the project wants to avoid any new N10 classifier decision, the safer fallback is Option A with transparent caveats and no strong "fluky scoring" language in the headline.

Do not redefine the whole analysis solely around yards per point. That would throw away exactly the defensive/return/explosive mechanisms the project set out to inspect. Also do not silently modify the N07 components; any component-level redesign should be a separately authorized methodology change.
