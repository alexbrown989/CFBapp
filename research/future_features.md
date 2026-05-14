# Future feature ideas (deferred from current notebooks)

This file tracks feature hypotheses that emerged during Phase 0 notebook
authoring but aren't part of the V5 `trigger_features` DDL or the
currently-active 02a-g candidate sets. Each item is one of:

- **Scheduled** — pencilled into a specific upcoming notebook (02c-g)
  as a candidate group, with a target notebook and any open research
  questions.
- **Deferred** — interesting but not currently slotted into any notebook;
  re-evaluate at the start of each new notebook plan or at the
  Phase 0 → Phase 1 boundary.
- **Rejected** — failed stability in a published notebook; preserved
  here with the rejecting commit hash and the feature-set version that
  failed.

This file is documentation only. It does NOT define schema, populate
`feature_validation.csv`, or feed N03 directly. Items move from
"Scheduled" into a notebook's `CANDIDATE_FEATURES` list at that
notebook's plan-approval step.

---

## Scheduled

### Momentum / decaying-shock features (target: 02c)

Hypothesis: an explosive play by the underdog has a decaying effect on
the favorite's near-future defensive performance — the closer in time
the prior underdog explosive play, the worse the favorite's next-drive
expected EPA-against. Two candidate operationalizations under
consideration:

  - `seconds_since_last_explosive_play_by_underdog`
  - next-drive outcome conditional on prior-drive explosive play (the
    drives-level version)

- **Target notebook:** 02c (explosive vs sustained). Will be incorporated
  into the 02c plan as a candidate feature group alongside the V5 DDL
  block-3 features. Plan-time decision on shape (linear / step /
  exponential / continuous-seconds-let-model-learn).
- **Open research question for 02c plan:** what does current literature
  say about post-explosive-play next-drive outcomes in CFB? To research
  before 02c plan write-up.

### Home-field advantage and neutral-site effects

- **Proposed by:** project owner (alexbrown989)
- **Proposed on:** 2026-05-12
- **Mechanism:** Home crowd noise affects offensive communication (snap
  counts, audibles) and visiting defensive coordination; neutral sites
  eliminate both, sometimes creating a third regime distinct from
  either home/away. Effect on comeback equity is direction-uncertain:
  home favorites may have more comeback capacity (crowd energy, no
  travel fatigue) but home underdogs may also be harder to put away.
- **Testable form:** at trigger time, two binary features:
  - fav_is_home (1 if favorite is playing at home, else 0)
  - is_neutral_site (1 if neutral-site venue, else 0)
- **Target notebook:** 02g (context — week, weather). Will be
  incorporated alongside the V5 DDL block-7 features.
- **Open research question for 02g plan:** does CFBD `/games`
  `neutralSite` flag and home/away team assignment carry enough info
  to derive both features reliably, or do older seasons have coverage
  gaps? To verify at 02g plan time from N00 audit data.

### Categorical-window momentum features (target: N03 or follow-up to N02c)

- **Proposed by:** project owner (alexbrown989)
- **Proposed on:** 2026-05-13
- **Origin:** N02c P3 trigger logic, after `seconds_since_last_dog_explosive_play`
  passed 2/3 Brier-improving folds under R6 stability — passes the
  rule's floor but doesn't strengthen across all test seasons
  (+0.00468 / +0.00780 / −0.00269) — and
  `prior_drive_had_dog_explosive_play` failed 0/3 stability with
  near-zero Brier deltas (−0.00009 / −0.00014 / −0.00010). See
  `research/corrections_log.md` for the chrono_key correction context.
- **Mechanism:** the corrected 02c results suggest momentum decay is
  finer-grained than possession boundaries (binary "prior drive had
  one" form failed) but the continuous-seconds signal is weak — only
  2 of 3 folds improve on Brier, with one fold going negative. A
  binned middle ground may capture what continuous-seconds catches in
  a more interpretable, less noise-prone form — explicit time windows
  give the model step changes the continuous signal had to earn
  through nonlinear transformation.
- **Inversion vs. literature:** Roebber 2022 found the binary
  streak-style momentum feature carried signal in NFL win-probability
  modelling; 02c's corrected result suggests this transfer doesn't
  hold for CFB comeback-trigger contexts. Possible explanation:
  trigger conditioning already selects for games where the underdog
  has been productive, compressing the variance the binary form needs
  to detect. The continuous form survives weakly because it preserves
  recency information the binary form discards.
- **Testable form:** three or more binary indicators based on
  time-since-last-dog-explosive-play, e.g.:
  - `had_dog_explosive_in_last_60s`
  - `had_dog_explosive_in_last_180s`
  - `had_dog_explosive_in_last_300s`
  - `had_dog_explosive_in_last_600s`
- **Open question:** bin boundaries are arbitrary; could test
  30/60/120/300/600 seconds or any reasonable schedule. Would benefit
  from a brief literature scan on momentum-decay timescales in
  play-by-play data.
- **Target notebook:** N03 if the validated 02c features show
  sufficient combined signal in production training to motivate
  refinement; otherwise deferred. Conditional on N03 model performance
  with the seven validated 02c features baseline.

---

## Deferred / not currently scheduled

(none yet)

---

## Rejected

(none yet — will populate from `validated_filters.json` `rejected_features`
at end of Phase 0)
