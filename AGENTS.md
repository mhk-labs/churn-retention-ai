# Instructions for coding assistants

Read README.md, docs/CONTRACTS.md and DECISIONS.md before changing the pipeline.
Keep changes focused on the requested task; preserve teammates' work.
Use feature branches and reviewable pull requests.
Prefer simple functions and explicit inputs/outputs over frameworks.
Keep imports free of downloads, training and paid API calls.
Never commit datasets, customer outputs, credentials, models or .env files.
Use synthetic fixtures for tests and demonstrations, and label them clearly.
Every feature must use only information available at its prediction cutoff.
Keep labels out of predictor columns and account for label maturity in time splits.
Do not present SHAP as causal evidence or simulated value as observed revenue saved.
Do not invent model metrics. Do not call paid APIs unless explicitly requested.
Run python -m unittest discover -s tests -v and python -m churn_retention.demo.
For app changes, also run Streamlit's AppTest when the app extra is installed.
Explain changes, validation and limitations so the author can explain the work.
