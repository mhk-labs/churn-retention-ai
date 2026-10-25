# EDA

`01_eda.ipynb` explores the source data using Python/pandas.
It previews the first five rows of all seven source tables, then analyses the two
label files and profiles, scans both transaction files and updated listening logs
in batches, and checks the first 100,000 original listening rows. It also compares
all pre-cutoff listening history with supplied later outcomes for a fixed
20,000-customer sample, using a working prediction date of 1 March 2017. Compare
7/28/90-day windows, all-history totals, recency, coverage and personal baselines.
Versions remain separate; the notebook does not select a modelling split or train a model.
Each short code cell has an explanation. Read it and run one cell at a time.

## Local setup

Activate your project environment first (`source .venv/bin/activate` on macOS).
From the repo root, install `python -m pip install -e ".[notebooks]"`.
Open `notebooks/01_eda.ipynb` in VS Code and choose the project's `.venv` Python
kernel. The first cell reports where Python is actually running.
Follow `data/README.md` to obtain your own KKBox access. Files can be CSVs in
`data/raw/` or the original `.7z` archives. Files required for full analysis are
extracted if no CSV is available, including
both transaction files and updated listening logs. The first archive extraction
can take several minutes and use several GB of cache space. The 30 GB original
listening history stays compressed. The source-quality section reads its first
100,000 rows; the later customer comparison streams the full archive once to
retain all sampled pre-cutoff history. The initial all-history scan can take around
15–20 minutes on this Mac. Matching reruns use the generated `cohort_cache/`, which
stores customer IDs, dates and durations without labels and must remain outside
Git. The all-history cache has a different filename and version from the earlier
February-only cache. Changing the comparison windows reuses retained history;
adding other raw listening columns would require a new source pass.
Archive reads use `libarchive-c`, which requires the system libarchive library
(available on this Mac and standard Kaggle runtimes). Large-file counts and
customer selection use pandas chunks, not SQL.

## Kaggle computation from VS Code

Keep editing **this local repo notebook**. A Kaggle notebook provides the remote
session and attached competition files. Its uploaded code is a generated snapshot;
do not edit an independent copy there and expect automatic file synchronisation.
Kaggle's Jupyter Server integration is experimental and sessions expire.

1. Install VS Code's Microsoft Jupyter extension.
2. Open your private Kaggle notebook with the KKBox competition attached.
3. In Kaggle, choose **Run > Kaggle Jupyter Server** and start the session if prompted.
4. Under **Manually Connect**, copy the **VS Code Compatible URL**.
5. Open `notebooks/01_eda.ipynb` in VS Code. Choose **Select Kernel > Select Another
   Kernel > Existing Jupyter Server**, enter the URL, and choose its Python kernel.
6. Run only the first cell. It must say **Python is running on: Kaggle**.

The server URL can contain an authentication token. Paste it directly into VS Code,
not into chat, this README, settings committed to Git, or a notebook cell. Keep the
notebook and its competition input private. You may need to reconnect after a
session expires. Save local notebook edits normally; Git tracks the local file.
Connecting a remote kernel is not file synchronisation with Kaggle's saved version.

Official references: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) and
[VS Code remote Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server).

## Prepare a launch snapshot from this source

After reviewing the notebook, run:

```bash
python scripts/prepare_kaggle_eda.py --owner YOUR_KAGGLE_USERNAME
kaggle kernels push -p data/processed/kaggle-eda --no-run
```

The preparation script stages only an output-cleared notebook and metadata under
ignored `data/processed/kaggle-eda/`. It attaches the official competition input,
keeps the notebook private, selects CPU, and enables Internet only so a missing
archive-reader dependency can be installed. No local data or credentials are copied.
`--no-run` saves a snapshot without executing cells. Omit it only to deliberately
validate the entire lesson in a saved Kaggle run. Regenerate the snapshot when
you want to save a new Kaggle version; day-to-day editing stays in VS Code.

## Outputs and scope

Local generated files go in ignored `data/processed/eda/`; Kaggle generated
files go in `/kaggle/working/churn-eda/`. Preview profile rows are not representative;
the original listening prefix is not representative either. Full-file results are
labelled explicitly. Figures cover transaction activity and updated daily
listening, plus pre-cutoff customer activity by outcome. Summaries, comparison
tables and daily aggregates are saved beside `summary.json`.
Clear notebook cell outputs before committing. Keep datasets, credentials and
customer outputs out of Git.

The active extraction cache is `data/processed/eda/cache/`; the sampled listening
history is in `data/processed/eda/cohort_cache/`. The latter avoids repeating the
full original-log scan. Original downloads stay in `data/raw/`; see
[`data/README.md`](../data/README.md) for the folder map. Obsolete SQL artifacts
and temporary verification snapshots are not part of the maintained workflow.

## Why these comparisons

The cutoff answers when a prediction is made; a history window answers how far
back a summary looks. Retain all available earlier history, then compare recent
7-day activity, four-week habits, broader 90-day context, and all-history coverage.
Lifetime totals need observation-length context: someone observed for two years
has more opportunity to accumulate activity than someone observed for one month.

Compare recorded minutes per day in the latest 28 days with the preceding 62 days.
Normalising by days makes the periods comparable; non-overlapping periods prevent
recent activity from appearing in its own baseline. A ratio of 1 means equal
recorded intensity, 0.5 means half as much, and 2 means twice as much.
Use this ratio only where earlier observation and quality conditions are met;
report its denominator. A zero baseline gives an undefined ratio.

Older records with no recent records and no earlier source records are separate
coverage groups. Neither a first log date nor missing window records establishes
actual customer tenure or uninterrupted observation. The chosen windows are EDA
candidates; later predictive selection belongs inside training/validation.

The customer comparison uses only original pre-cutoff sources, keeps labels
separate until analysis, reports coverage, excludes invalid or duplicated daily
listening records, and marks conflicting latest-day renewal flags as ambiguous.
Read sections 14–19 for the definitions and denominators. These are retrospective
associations within the sample, not model metrics or causal claims.

Confirm the working prediction date, source availability, eligible population,
outcome periods and label maturity together before building features or models.
`02_features.ipynb` has not been created.
