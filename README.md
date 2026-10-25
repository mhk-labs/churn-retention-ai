# Churn Retention AI

Predict subscription churn, explain model predictions, and support retention decisions.

**Status: project skeleton.** No model has been trained, no real customer data is included,
and no business impact has been measured. The runnable demo uses synthetic fixtures only.

Exploration is available in [the EDA notebook](notebooks/01_eda.ipynb). It inspects
KKBox sources and compares pre-cutoff customer history with supplied later outcomes.
See [notebook setup](notebooks/README.md) and [the data folder map](data/README.md).
The training, SHAP and LLM pipelines remain unimplemented.

## Start here

Use Python 3.11 or 3.12. From a terminal:

```bash
git clone https://github.com/mhk-labs/churn-retention-ai.git
cd churn-retention-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,dev]"
python -m churn_retention.demo
python -m unittest discover -s tests -v
python -m streamlit run app/streamlit_app.py
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.
For modelling, additionally run `python -m pip install -e ".[model,notebooks]"`.
The CLI demo and core tests need only `python -m pip install -e .`.
Dependency ranges are starter constraints, not a lockfile; lock the agreed environment before
publishing model results.

## Understand and contribute

- [Project explained](docs/PROJECT_GUIDE.md): what each stage does, with a worked example.
- [Team workflow](CONTRIBUTING.md): branches, reviews and using AI coding tools.
- [Interfaces](docs/CONTRACTS.md): proposed handoff between the three workstreams.
- [First tasks](docs/BACKLOG.md): small tasks with acceptance criteria.
- [Decision log](DECISIONS.md): agreed choices and unresolved questions.

| Location | Purpose |
| --- | --- |
| `data/` | Download instructions; raw and processed data stay local |
| `notebooks/` | Numbered exploration, then move reusable code into src |
| `src/churn_retention/` | Features, training, evaluation, explanations, retention and LLM modules |
| `app/` | Streamlit UI |
| `tests/` | Offline checks; extend with leakage and schema tests |
| `reports/figures/` | Reviewed charts for the README |
| `artifacts/` | Local model outputs, ignored by Git |
| `docs/` | Project guide, interfaces and starter backlog |

## Planned modelling approach

KKBox is the main dataset; IBM Telco is an optional short warm-up.
The plan's working churn definition is no valid renewal within 30 days after expiry.
Verify the competition label construction before implementing it.
Choose prediction dates, eligible customers, feature cutoffs, and label availability
before training. Compare logistic regression with LightGBM on a defensible temporal holdout.
Measure PR-AUC, ROC-AUC, log loss, and recall/precision at the contact budget.

SHAP describes contributions to a prediction, not proven causes of churn.
Retention value is a scenario estimate, not demonstrated causal revenue saved.
The LLM will use supplied evidence and an approved action menu; it will not send messages.

## Team

Roles are unassigned. Choose owners together; each person reviews another person's PR.
See the backlog for parallel starting points. Deployment, production data pipelines, model training,
SHAP, budget simulation and an API-backed assistant are future work.
