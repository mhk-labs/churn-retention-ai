# Decisions

## 2026-10-24: starter conventions
- Python package under src/churn_retention; lightweight core and optional dependencies.
- Offline synthetic demo first, with no paid API calls.
- Precomputed predictions/explanations planned for the app.
- Roles remain unassigned.
- Main changes go through peer-reviewed PRs (admin enforcement still to configure).

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
