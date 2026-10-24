# Proposed interfaces

Agree changes together before integrating. These are a starter contract, not a fixed feature list.

## Feature table
One row per (customer_id, prediction_date); prediction_date is an ISO date interpreted
as the start of that day. Source events must be strictly earlier and available by then.
Columns: customer_id, prediction_date, numeric/categorical feature columns.
Keep the label and its availability date in a separate keyed table.
Reject duplicate keys; record missing-value policy and source windows.
A training label must be available before the simulated model-fit date.

## Prediction record (see demo.py for a synthetic example)
- customer_id: string
- prediction_date: ISO date
- churn_probability: float in [0, 1]
- model_version: string
- is_synthetic: boolean
- drivers: list of objects with feature, value, direction (increases_risk/decreases_risk),
  and description

The fixture has illustrative drivers, not computed SHAP values.
The future explanation implementation must also record SHAP values, base value and
output scale (probability or raw/log-odds), consistently with its model/explainer.

## Assistant result (future)
customer_id, prediction_date, action_id from an agreed menu, evidence referencing supplied
features, draft_message, needs_review, generation_method and prompt/model version.
Validate the schema and unsupported factual claims. Default to human review.
The starter app displays a static template labelled as such, not LLM output.

## Artifacts
Features and prediction tables go in ignored data/processed; model files and metadata in
ignored artifacts/. Keep only small synthetic examples in code/tests.
Metadata should record data versions, sampling seed, cutoffs, feature list, model version,
dependency lock and evaluation settings.
