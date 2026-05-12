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

---

## Deferred / not currently scheduled

(none yet)

---

## Rejected

(none yet — will populate from `validated_filters.json` `rejected_features`
at end of Phase 0)
