# Churn Retention AI

A three-person project exploring subscription churn with KKBox data using Python
and pandas. **We are doing EDA. No model, SHAP or LLM pipeline is implemented.**

## Start here

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
```

1. Join the [KKBox competition](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data)
   with your own Kaggle account and accept its data terms.
2. Download the seven source archives into `data/raw/`: `train`, `train_v2`,
   `members_v3`, `transactions`, `transactions_v2`, `user_logs` and `user_logs_v2`
   (each named `*.csv.7z`). Extracted CSV inputs also work.
3. Open [01_eda.ipynb](notebooks/01_eda.ipynb) in VS Code, select the `.venv`
   Python kernel and run cells in order. Markdown explains each step.
4. Run [02_eda_cache.ipynb](notebooks/02_eda_cache.ipynb) to inspect the listening
   distributions using notebook 01's cache and extracted labels.

The first notebook streams the original listening archive, avoiding a 30 GB
extraction. Building the sampled history previously took about 15–20 minutes on
this Mac. Keep the cache for faster reruns. Other source files are extracted when
needed and use several GB. Archive streaming requires the system libarchive
library, available on this Mac and standard Kaggle runtimes.

## Where things live

| Folder | What belongs here |
| --- | --- |
| `notebooks/` | EDA notebooks, with explanations beside the code |
| `data/raw/` | Original downloads and their provenance records |
| `data/processed/` | Generated summaries, figures, extracted CSV cache and sampled history |
| `churn_retention/` | Reusable Python code; currently the synthetic demo and optional upload helper |
| `tests/` | Small synthetic checks that do not need competition data |

Keep `data/processed/cohort_cache/` for the sampled listening history and manifest,
and an extracted `train_v2.csv` in `data/raw/` for notebook 02. Notebook 01 recreates
`cache/` and its summaries as needed; notebook 02 recreates `02_cache/` charts.
The optional upload helper recreates `kaggle-eda/`. These generated files can be
deleted when no longer needed. Downloaded data and customer-level notebook outputs
stay local/private and out of Git.
For an older checkout, move existing `data/processed/eda/` contents up to
`data/processed/` once; avoid overwriting files if both locations contain results.

`README.md` is the project guide. `AGENTS.md` contains coding-assistant rules;
`CLAUDE.md` points Claude to those same rules. Add folders and modules when there
is working code that needs them.

## Which data files we use

These are the current EDA sources and the starting point for the feature prototype:

**The sample's customer IDs come from `train_v2.csv`; its saved listening rows
come from `user_logs.csv` (without `_v2`). Transaction summaries use
`transactions.csv` (without `_v2`) for the same IDs.** We are not using an
all-v2 bundle: each file has its own release suffix. `members_v3.csv` is currently
used only for profile checks, not the saved sample or customer-history features.

| File in `data/raw/` (CSV or `.csv.7z`) | Current use |
| --- | --- |
| `train_v2.csv` | Supplied outcomes and customer pool: **970,960 customers**; select the EDA sample from these IDs |
| `transactions.csv` | Original subscription/payment events; summarise earlier events for the sampled customers |
| `user_logs.csv` | Original listening history; retain earlier rows for those same customers |
| `members_v3.csv` | Inspect profile quality; not yet joined into the customer comparison or used as predictors |
| `train.csv` | Compare label releases; keep separate from `train_v2.csv` |
| `transactions_v2.csv` | Inspect updated-source coverage/quality; excluded from current customer-history features |
| `user_logs_v2.csv` | Inspect updated-source coverage/quality; excluded from current customer-history features |

Use the supplied **`is_churn`** target: `1` means churn and `0` means renewal.
`is_cancel` describes a transaction event, not that target. The
[competition host](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/43622)
defines churn around failure to renew within 30 days after expiry. The local
provider reference, `data/raw/WSDMChurnLabeller.scala`, treats a renewal gap of
**30 days or more** as churn and contains cancellation/tie-handling rules.
We have not reproduced supplied labels or verified each release's outcome period.

## Sample or full data?

**Use the 20,000-customer sample for EDA and the first feature/model prototype.
Plan final experiments on all eligible customers in the agreed label cohorts.**
For the current `train_v2.csv` pool that means up to 970,960 customers, subject to
the eligibility and timing checks below. It does not mean treating every log row
or every member profile as a separate training example. Build one customer/date
feature row by processing the source histories in batches.

Notebook 01 sections **14–15** generate the sample:

1. Select only the `msno` column from the full `train_v2.csv`.
2. Randomly choose **20,000 distinct IDs**, without replacement, using seed **42**.
   This is customer sampling, without using or balancing churn labels. The same
   input file/order and seed reproduce the same IDs.
3. Stream all **392,106,543** original listening rows. Keep rows for those IDs
   dated before **1 March 2017**, retaining `msno`, `date` and `total_secs`.
4. Save and reuse this history; summarise original transactions for the same IDs
   separately. Join supplied labels after computing history summaries.

The saved listening sample is
**`data/processed/cohort_cache/precutoff_listening_sample.csv`** (~309 MB).
It contains **4,911,296 rows**, covering **1 January 2015–28 February 2017**.
**17,521** sampled customers have records; **2,479** have none and remain in the
20,000-customer comparison. Reconstruct those IDs from the label file and seed;
do not reconstruct the cohort from only customers present in the listening CSV.

Beside it, `precutoff_listening_manifest.json` records the source fingerprint,
seed, cutoff, row counts and selected-ID hash. Notebook 01 verifies these before
reusing the cache; notebook 02 recreates the same customer selection. This cache
is filtered source history, not a complete modelling dataset: it has no labels,
profiles or transactions, and quality rules apply when summaries are calculated.

Notebook 01 also performs full-file source audits for labels, profiles,
transactions and updated listening logs. Only the original-log quality preview
is a 100,000-row prefix; that prefix is separate from the customer-history sample.

## Current analysis choices

- Use the start of **1 March 2017** as the working prediction cutoff. Customer
  summaries use only original `user_logs.csv` and `transactions.csv` events
  strictly before that date. Keep updated releases separate until reconciled.
- Retain all earlier listening history, then compare **7-, 28- and 90-day**
  activity, recency, coverage and all-history totals. The cache stores IDs, dates
  and durations; adding other fields requires another source scan.
- Compare recent 28-day minutes per day with the preceding 62-day rate. Require
  a first valid observation at least 90 days earlier, no unusable records in that
  period and a positive baseline. Only in this comparison do absent window rows
  count as zero recorded minutes; general duration totals remain missing.
- Exclude invalid durations and all duplicate customer/day rows from listening
  summaries. Mark conflicting latest-day renewal flags as ambiguous. Missing
  records do not prove inactivity; observation span is not verified tenure.

These are retrospective EDA choices, not a final modelling design. Notebook 01
contains source-quality checks and definitions; notebook 02 explores distributions.
Visible group differences are associations, not model metrics or causal evidence.

## What is settled, and what comes next

The file roles, target flag, reproducible development sample, history windows and
missing-history distinction are settled for EDA. Keep the existing listening and
renewal summaries as candidate features. Do not drop customers merely because
profiles/history are absent, or interpret unusual payment plans as errors without
checking their meaning.

**Still open before fitting a model:** verify which expiry/outcome period each
label release represents, who was eligible at prediction time, and which fields
were actually available then. The provider's example uses a 31 January history
cutoff and February expiry candidates; it does not establish that our 1 March EDA
cutoff is appropriate for `train_v2.csv`. A feature can precede our chosen cutoff
yet already reveal part of its customer's outcome. A training label also needs
its renewal-observation window to finish before the simulated model-fit date.

1. Resolve that timeline and choose training/validation/test cohorts. Do not
   automatically use `train.csv` to train and `train_v2.csv` to test without
   checking label maturity. Keep updated histories separate unless reconciliation
   is needed for the agreed dates.
2. Build features: **one row per `(customer_id, prediction_date)`**, using only
   information available before the start of that day. Reject duplicate keys,
   keep labels and their availability dates separate, and document missing values.
3. Train a logistic-regression baseline with mature labels and a justified time
   split. Fit preprocessing on training data; reserve a final test set. Evaluate
   ranking, probability quality and precision/recall at an agreed contact budget.
4. Compare LightGBM, then add explanations and retention support if useful.
   SHAP is not causal evidence; simulated value is not observed revenue saved.

Future prediction records should include customer/date keys, churn probability,
model version, a synthetic-data flag and drivers. Computed SHAP explanations must
record their base value and output scale. Any future LLM uses supplied evidence
and an agreed action menu, with schema checks, human review and an explicit cost
limit. No paid API calls are part of the current project.

## Working together

Choose one focused task and owner. Pull `main`, create a branch such as
`feature/listening-features`, and open a PR for a teammate to review before merging.
Use `fix/` for corrections and `chore/` for maintenance. Explain the code and record
agreed analysis changes here. Branch protection still needs to be configured.

```bash
python -m pip install -e ".[notebooks,dev]"
python -m unittest discover -s tests -v
python -m churn_retention.demo
python -m ruff check .
```

Tests cover temporal summaries, synthetic demo records and safe upload preparation.
The app test runs when Streamlit is installed. Never commit datasets, customer
outputs, credentials or models. Clear notebook outputs before committing. Keep
source versions, sampling, cutoffs, feature lists and evaluation settings with
local generated results; lock dependencies before publishing model results.

## Optional tools

The synthetic demo is independent of EDA and uses two fictional customers:

```bash
python -m pip install -e ".[app]"
python -m streamlit run churn_retention/app.py
```

Notebook 01 can also run on Kaggle with the official competition input attached.
Its first cell reports the runtime: Kaggle uses `/kaggle/input` and saves under
`/kaggle/working/churn-eda`; local execution uses the `data/` folders above.
Notebook 02 currently runs locally. Kaggle outputs do not copy back automatically.

To prepare a private Kaggle copy of notebook 01 with cleared outputs:

```bash
python -m churn_retention.kaggle --owner YOUR_KAGGLE_USERNAME
kaggle kernels push -p data/processed/kaggle-eda --no-run
```

The first command stages only notebook code and metadata; the second requires
Kaggle CLI access and saves without running. GitHub and Kaggle saved notebooks
are separate copies: pushing Git does not update Kaggle.
