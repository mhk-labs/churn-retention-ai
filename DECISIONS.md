# Decisions

## 2026-10-24: starter conventions
- Python package under src/churn_retention; lightweight core and optional dependencies.
- Offline synthetic demo first, with no paid API calls.
- Precomputed predictions/explanations planned for the app.
- Roles remain unassigned.
- Main changes go through peer-reviewed PRs (admin enforcement still to configure).

## 2026-10-25: working assumptions for outcome-group EDA
- Compare a fixed random sample of up to 20,000 customer IDs from train_v2.csv,
  seed 42. Select IDs without using outcomes; join labels only after summarising history.
- Use the start of 2017-03-01 as the working prediction moment. Listening covers
  2017-02-01 through 2017-02-28, with equal 14-day halves for the change comparison.
- Use original user_logs.csv and transactions.csv only. Exclude events on or after
  the cutoff; do not combine updated releases before reconciling them.
- Stream the original listening archive once and cache only the selected customer
  history under ignored data/processed. Cache metadata records the source stat,
  dates, seed and hash of selected IDs.
- Exclude all duplicate customer/day listening rows and invalid durations from this
  comparison. Report exclusions and history coverage. Keep absent history distinct
  from recorded activity; compare changes where both halves have valid records.
- Mark conflicting latest-day auto-renew flags as ambiguous. Do not interpret the
  latest event flag as a verified current subscription state.
- These choices support retrospective EDA only. Event dates do not prove ingestion
  availability, and supplied-label cohort eligibility at the cutoff is unverified.
  Final model population, label maturity and temporal splits remain open.

## 2026-10-25: extend EDA to all available pre-cutoff history
- Supersede the February-only listening selection above. Retain all available
  original-source listening records before 2017-03-01 for the same sampled IDs.
  Keep the cutoff, sample size and seed fixed so the history comparisons are comparable.
- Cache IDs, dates and durations under a distinct all-history cache version. A
  lookback-window change must not require another raw-source scan. Other listening
  columns are not retained by this activity cache.
- Compare 7-, 28- and 90-day activity plus all-history totals, observed history span,
  recency, and recent/older-only/absent source coverage. An observed history span is
  not verified tenure or proof of continuous observation.
- Compare recorded minutes per day in the latest 28 days with the preceding 62 days.
  Require a first valid observation at least 90 days earlier, no unusable records in
  that period and a positive baseline. Report the available-customer denominator.
- Treat absent window rows as zero recorded minutes only in that explicitly defined
  baseline comparison; retain missing duration totals in the general window table.
- Windows are interpretable EDA candidates, not established best model settings.
  Final eligibility, source availability, label maturity and model evaluation remain open.

## Must resolve before modelling
- Exact KKBox file versions and label construction.
- Eligible customer population and prediction moment relative to subscription expiry.
- Feature windows and data availability cutoff.
- Which temporal cohorts have mature labels available at each simulated training date.
- Validation/test dates and handling of customers repeated across months.
- Deterministic customer sampling method and seed.
- Metrics, contact budget, value horizon, offer/contact costs and success assumptions.
- Approved retention actions, provider, spending limits, cache and evaluation examples.

## Decision template
Date / question / choice / evidence / trade-off / owner / issue or PR.
