"""Prepare and evaluate the March sample as a retrospective experiment."""

import hashlib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_retention.features import FEATURE_COLUMNS

PREDICTION_DATE = "2017-03-01"
LABEL_SOURCE = "train_v2.csv"
SPLIT_SEED = 42
CATEGORICAL = ["latest_auto_renew_status", "history_coverage"]
NUMERIC = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL]


def prepare_sample(features, labels, metadata):
    """Check the notebook 03 exports and return predictors and labels by customer ID.

    This accepts prototype exports for same-month evaluation only. It does not
    establish point-in-time ingestion availability or a historical model-fit date.
    """
    if set(features.columns) != {"customer_id", "prediction_date", *FEATURE_COLUMNS}:
        raise ValueError("Feature export must contain only the two keys and the 15 candidate features.")
    if set(labels.columns) != {"customer_id", "is_churn", "label_source"}:
        raise ValueError("Expected the separate supplied-label export from notebook 03.")
    for table in (features, labels):
        ids = table["customer_id"]
        if ids.isna().any() or not ids.is_unique or ids.astype(str).str.strip().eq("").any():
            raise ValueError("Each export must have unique, non-empty customer IDs.")
    if set(features["customer_id"]) != set(labels["customer_id"]):
        raise ValueError("Feature and label customer IDs must match exactly.")
    if not labels["label_source"].eq(LABEL_SOURCE).fillna(False).all():
        raise ValueError("Use only the supplied train_v2.csv labels for this March experiment.")
    if not labels["is_churn"].isin([0, 1]).all() or labels["is_churn"].nunique() != 2:
        raise ValueError("Expected binary outcomes with both classes present.")
    if not features["prediction_date"].eq(PREDICTION_DATE).fillna(False).all():
        raise ValueError("Feature rows must use the 1 March 2017 prediction date.")
    ids_hash = hashlib.sha256("\n".join(sorted(features["customer_id"])).encode()).hexdigest()
    if (metadata.get("prediction_date") != PREDICTION_DATE
            or metadata.get("sample_size") != len(features)
            or metadata.get("sample_ids_sha256") != ids_hash
            or metadata.get("feature_columns") != list(FEATURE_COLUMNS)):
        raise ValueError("Feature export does not match its sample, date or column metadata.")
    sources = metadata.get("sources", {})
    listening = sources.get("listening_source_fingerprint", {})
    if (Path(sources.get("customer_ids_and_supplied_labels", "")).name != LABEL_SOURCE
            or Path(sources.get("transactions", "")).name not in
            {"transactions.csv", "transactions.csv.7z"}
            or Path(listening.get("source", "")).name not in {"user_logs.csv", "user_logs.csv.7z"}
            or listening.get("cutoff") != PREDICTION_DATE + "T00:00:00"
            or listening.get("sample_ids_sha256") != ids_hash):
        raise ValueError("Expected the original transaction and listening sources for this sample.")

    predictors = features.set_index("customer_id").sort_index()[list(FEATURE_COLUMNS)].copy()
    predictors[NUMERIC] = predictors[NUMERIC].astype(float)
    if predictors[NUMERIC].isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Infinite feature values are not supported.")
    categories = predictors[CATEGORICAL].astype(object)
    predictors[CATEGORICAL] = categories.where(categories.notna(), float("nan"))
    outcomes = labels.set_index("customer_id")["is_churn"].reindex(predictors.index).astype(int)
    return predictors, outcomes


def split_customers(outcomes, seed=SPLIT_SEED):
    """Return disjoint customer IDs in a reproducible, stratified 60/20/20 split."""
    if not outcomes.index.is_unique:
        raise ValueError("Split one outcome per customer.")
    outcomes = outcomes.sort_index()
    train, remaining = train_test_split(
        outcomes, test_size=0.4, stratify=outcomes, random_state=seed,
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, stratify=remaining, random_state=seed,
    )
    return {"train": train.index, "validation": validation.index, "test": test.index}


def logistic_baseline():
    """Build an unfitted baseline; fit imputation, scaling and encoding on train only."""
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        ), NUMERIC),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="Missing"),
            OneHotEncoder(handle_unknown="ignore"),
        ), CATEGORICAL),
    ])
    return make_pipeline(preprocess, LogisticRegression(C=1.0, max_iter=1000))


def probability_metrics(outcomes, probabilities):
    """Score probability quality and ranking against binary supplied outcomes."""
    return {
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(outcomes, probabilities)),
        "roc_auc": float(roc_auc_score(outcomes, probabilities)),
    }
