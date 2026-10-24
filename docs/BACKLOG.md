# Starter backlog

These are task descriptions, not automatically created GitHub issues. Owners TBD.

| Task | Done when |
| --- | --- |
| Agree prediction problem | Population, prediction moment, label window and success metrics recorded |
| Inspect KKBox schema and versions | File inventory, date ranges and availability assumptions documented |
| Agree temporal split | Cohorts, label maturity and fit dates justified with a small timeline |
| Download and sample | Documented deterministic customer sample; same IDs across source tables |
| Build first features | One row per key; cutoff, duplicates and missing-value tests pass |
| Logistic regression baseline | Train-only preprocessing; holdout metrics and budget comparison saved |
| LightGBM and SHAP | Same split; comparison with baseline; explanation scale documented |
| Retention scenarios | Explicit costs, value horizon and success assumptions; random comparator |
| LLM assistant | Approved actions, validated output, cache, cost limit and 20 evaluation cases |
| Connect app | Replaces fixtures with versioned outputs; no training at startup |
| Deploy and document | Safe demo data; live URL; reproducible environment; limitations and demo video |

Parallel start: data owner inspects schema, model owner designs evaluation on synthetic
inputs, app owner works from the fixture contract. Do not train on real data until
the temporal problem is agreed.
