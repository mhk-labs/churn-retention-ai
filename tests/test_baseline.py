"""Synthetic checks for the retrospective model's inputs and separation of customers."""

import hashlib
import importlib.util
import unittest

try:
    import pandas as pd
except ImportError:
    pd = None

HAS_MODEL = pd is not None and importlib.util.find_spec("sklearn") is not None
if HAS_MODEL:
    from churn_retention.baseline import (
        NUMERIC,
        logistic_baseline,
        prepare_sample,
        probability_metrics,
        split_customers,
    )
    from churn_retention.features import FEATURE_COLUMNS


@unittest.skipUnless(HAS_MODEL, "Install the model extra for baseline tests.")
class BaselineTests(unittest.TestCase):
    def sample(self):
        ids = [f"synthetic-{i:03}" for i in range(100)]
        features = pd.DataFrame({name: [float(i % 5) for i in range(100)] for name in NUMERIC})
        features["latest_auto_renew_status"] = "synthetic-renewal"
        features["history_coverage"] = "synthetic-coverage"
        features["customer_id"] = ids
        features["prediction_date"] = "2017-03-01"
        labels = pd.DataFrame({"customer_id": ids[::-1], "is_churn": [1, 0, 0, 0, 0] * 20,
                               "label_source": "train_v2.csv"})
        ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        metadata = {
            "prediction_date": "2017-03-01", "sample_size": 100,
            "sample_ids_sha256": ids_hash, "feature_columns": list(FEATURE_COLUMNS),
            "sources": {"customer_ids_and_supplied_labels": "data/raw/train_v2.csv",
                        "transactions": "data/raw/transactions.csv.7z",
                        "listening_source_fingerprint": {
                            "source": "data/raw/user_logs.csv.7z",
                            "cutoff": "2017-03-01T00:00:00", "sample_ids_sha256": ids_hash,
                        }},
        }
        return features, labels, metadata

    def test_labels_align_by_key_and_only_features_become_predictors(self):
        X, y = prepare_sample(*self.sample())
        self.assertEqual(X.columns.tolist(), list(FEATURE_COLUMNS))
        self.assertTrue(X.index.equals(y.index))
        self.assertEqual(y.loc["synthetic-099"], 1)
        self.assertEqual(y.loc["synthetic-000"], 0)

    def test_rejects_duplicate_keys_mismatched_ids_and_wrong_source_or_date(self):
        features, labels, metadata = self.sample()
        duplicate = pd.concat([features, features.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_sample(duplicate, labels, metadata)
        with self.assertRaisesRegex(ValueError, "match exactly"):
            prepare_sample(features, labels.iloc[:-1], metadata)
        with self.assertRaisesRegex(ValueError, "train_v2"):
            prepare_sample(features, labels.assign(label_source="train.csv"), metadata)
        with self.assertRaisesRegex(ValueError, "1 March"):
            prepare_sample(features.assign(prediction_date="2017-04-01"), labels, metadata)
        with self.assertRaisesRegex(ValueError, "original transaction"):
            wrong = {**metadata, "sources": {**metadata["sources"],
                                             "transactions": "transactions_v2.csv"}}
            prepare_sample(features, labels, wrong)
        with self.assertRaisesRegex(ValueError, "only the two keys"):
            prepare_sample(features.assign(is_churn=1), labels, metadata)

    def test_split_is_repeatable_stratified_and_disjoint(self):
        _, y = prepare_sample(*self.sample())
        splits = split_customers(y)
        reversed_splits = split_customers(y.iloc[::-1])
        all_ids = []
        for name, expected in [("train", 60), ("validation", 20), ("test", 20)]:
            ids = splits[name]
            self.assertTrue(ids.equals(reversed_splits[name]))
            self.assertEqual(len(ids), expected)
            self.assertEqual(y.loc[ids].sum(), expected // 5)
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(y.index))

    def test_model_handles_missing_values_and_unseen_categories_without_refitting(self):
        X, y = prepare_sample(*self.sample())
        splits = split_customers(y)
        train = X.loc[splits["train"]].copy()
        train.loc[train.index[0], "recorded_minutes_28d"] = float("nan")
        model = logistic_baseline().fit(train, y.loc[train.index])
        unseen = X.loc[splits["test"]].copy()
        unseen["latest_auto_renew_status"] = "synthetic-held-out-category"
        unseen["recorded_minutes_28d"] = float("nan")
        encoder = model[0].named_transformers_["categorical"][-1]
        probabilities = model.predict_proba(unseen)[:, 1]
        self.assertNotIn("synthetic-held-out-category", encoder.categories_[0])
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        metrics = probability_metrics(y.loc[unseen.index], [0.2] * len(unseen))
        self.assertAlmostEqual(metrics["average_precision"], 0.2)
        self.assertAlmostEqual(metrics["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
