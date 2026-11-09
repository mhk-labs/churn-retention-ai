# Churn Retention AI

A three-person project exploring subscription churn with KKBox data using Python
and pandas. **EDA, customer features and a first logistic regression baseline
are available.** The baseline measures performance on held-out customers from
the same March cohort. SHAP and the LLM pipeline remain future work.

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
   (each named `*.csv.7z`). Extracted CSV inputs also work. The ~7 GB `user_logs`
   archive is only needed to rebuild the listening sample in notebook 01 (locally or
   on Kaggle); notebooks 02–03 use the saved `cohort_cache/` instead.
3. Open [01_eda.ipynb](notebooks/01_eda.ipynb) in VS Code, select the `.venv`
   Python kernel and run cells in order. Markdown explains each step.
4. Run [02_eda_cache.ipynb](notebooks/02_eda_cache.ipynb) to inspect the listening
   distributions using notebook 01's cache and extracted labels.
5. Run [03_features.ipynb](notebooks/03_features.ipynb) to build the 20,000-customer
   feature prototype. It starts with a synthetic example and keeps supplied labels
   separate.
6. Review [04_label_timeline.ipynb](notebooks/04_label_timeline.ipynb) for the label
   timeline. Rerun it only when rechecking the audit; it reads both transaction
   releases to rebuild labels with the provider's rules.
7. Install `python -m pip install -e ".[model]"` and run
   [05_baseline.ipynb](notebooks/05_baseline.ipynb). It reuses notebook 03's saved
   exports, so no raw data scan or notebook 04 rerun is needed for this baseline.

The first notebook streams the original listening archive, avoiding a 30 GB
extraction. Building the sampled history previously took about 15–20 minutes on
this Mac. Keep the cache for faster reruns. Other source files are extracted when
needed and use several GB. Archive streaming requires the system libarchive
library, available on this Mac and standard Kaggle runtimes.

## Where things live

| Folder | What belongs here |
| --- | --- |
| `notebooks/` | EDA and feature notebooks, with explanations beside the code |
| `data/raw/` | Original downloads and their provenance records |
| `data/processed/` | Generated summaries, figures, extracted CSV cache and sampled history |
| `churn_retention/` | Customer features, label rules, baseline model and optional tools |
| `tests/` | Small synthetic checks that do not need competition data |

Keep `data/processed/cohort_cache/` for the sampled listening history and manifest,
and an extracted `train_v2.csv` in `data/raw/` for notebook 02. Notebook 01 recreates
`cache/` and its summaries as needed; notebook 02 recreates `02_cache/` charts.
The optional upload helper recreates `kaggle-eda/`. These generated files can be
deleted when no longer needed. Downloaded data and customer-level notebook outputs
stay local/private and out of Git.
Notebook 03 saves `customer_features.csv`, `supplied_labels_sample.csv` and
`feature_metadata.json` directly under `data/processed/`. It streams original
transactions without extracting another full CSV. Its metadata explicitly marks
the output as **not ready for training** in a historical simulation. Notebook 05
uses these exports for the retrospective experiment described below.
It saves `baseline_split.csv` and aggregate `baseline_metrics.json` under
`data/processed/`; keep the split fixed for later model comparisons.
Notebook 03 section 5.2 compares each full source file's earliest/latest dates with
the saved sample. The original downloads are unchanged; only the sample applies
its customer/date filters. Keep `source_date_ranges.json` to avoid repeating the
full-source date scans; deleting it requires another audit of the large archives.
The full `user_logs.csv` range comes from the sample manifest: notebook 01 records
it during the scan that builds the sample, so notebook 03 never rescans that file.
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
| `train_v2.csv` | Supplied outcomes for **March 2017 expiries** (notebook 04) and customer pool: **970,960 customers**; select the EDA sample from these IDs |
| `transactions.csv` | Original subscription/payment events; summarise earlier events for the sampled customers |
| `user_logs.csv` | Original listening history; retain earlier rows for those same customers |
| `members_v3.csv` | Inspect profile quality; not yet joined into the customer comparison or used as predictors |
| `train.csv` | Supplied outcomes for **February 2017 expiries** (notebook 04); not used in the sample or model for now |
| `transactions_v2.csv` | Adds March 2017 events (needed to rebuild labels in notebook 04) and ~290k earlier events missing from the original; excluded from current customer-history features |
| `user_logs_v2.csv` | Inspect updated-source coverage/quality; excluded from current customer-history features |
| `sample_submission_v2.csv` | Competition test-customer IDs and prediction-file template; not downloaded or used in the current prototype |
| `sample_submission_zero.csv` | Older competition submission template; not used in the current prototype |

The competition's submission files are separate from our 20,000-customer sample.
Their prediction column is a placeholder to replace with model probabilities,
not ground-truth labels. They must not be used as labelled training data.
Notebook 05 splits the 20,000-customer sample into local training, validation and
test sets. The March activity files are source history, not a labelled test set.

For the current 1 March prototype, keep history strictly before 1 March. An
April prediction would move the cutoff to 1 April and could use March history
for the appropriate target customers. Updated transaction releases overlap, so
reconcile them before combining. This is a proposed forecast setup; the existing
label-alignment and eligibility checks below still apply.

Use the supplied **`is_churn`** target: `1` means churn and `0` means renewal.
`is_cancel` describes a transaction event, not that target. The
[competition host](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/43622)
defines churn around failure to renew within 30 days after expiry. The local
provider reference, `data/raw/WSDMChurnLabeller.scala`, treats a renewal gap of
**30 days or more** as churn and contains cancellation/tie-handling rules.
Notebook 04 reproduces these rules: most rebuilt labels match, but not every one.

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
missing-history distinction are settled for the prototype. Notebook 03 uses 15
candidate features from the existing listening and renewal summaries, with an
explicit predictor list in `churn_retention/features.py`. Do not drop customers
merely because profiles/history are absent, or interpret unusual payment plans as
errors without checking their meaning.

**Label timeline (notebook 04).** Rebuilding labels with the provider's rules
(`churn_retention/labels.py`) matches **`train.csv` to February 2017 expiries**
(January history; kappa 0.84) and **`train_v2.csv` to March 2017 expiries**
(February history; kappa 0.66). The March match is weaker partly because renewals
can fall in April, after the data ends. Each file's prediction date is the start
of its expiry month: **1 February for `train.csv`, 1 March for `train_v2.csv`**.
This supports using notebook 03's 1 March cutoff with `train_v2.csv` for the
baseline. For `train.csv`, 94% of the matched February customers in the audit
already show a renewal before 1 March.

**First baseline:** use `train_v2.csv` only, with the existing 20,000 customers and
1 March features from the original listening and transaction histories. Split
by customer with seed 42: **12,000 train / 4,000 validation / 4,000 test**.
Stratification preserves the churn proportion. Fit imputation, scaling, category
encoding and logistic regression on the training group. Compare log loss,
average precision and ROC AUC with the constant training churn-rate reference.

This experiment uses supplied outcomes after they are known. It does not claim
a fitted model was available on 1 March. Earlier EDA informed feature choices on
this cohort, so the test is an exploratory check on held-out customers from the
same month. Some supplied labels cannot be rebuilt with the available history.
Forward evaluation needs a later labelled cohort with observed outcomes and a fit
date after the earlier training labels mature. `train.csv` is an earlier cohort.

Next, compare renewal-only features with the listening additions on validation.
Keep the test split fixed and reserve it for reporting. Verify source availability
before adding the extra pre-March events in `transactions_v2.csv`. Then consider
LightGBM and explanations. SHAP is not causal evidence; simulated value is not
observed revenue saved.

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
python -m pip install -e ".[notebooks,dev,model]"
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
