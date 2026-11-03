"""Synthetic checks for customer features, unknown history and temporal boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pandas as pd
except ImportError:
    pd = None
else:
    from churn_retention.features import (
        FEATURE_COLUMNS,
        build_features,
        source_date_ranges,
        transaction_chunks,
    )


@unittest.skipIf(pd is None, "Install pandas to check feature construction.")
class FeatureTests(unittest.TestCase):
    cutoff = "2017-03-01"

    def history(self, rows):
        return pd.DataFrame(rows, columns=["msno", "date", "total_secs"])

    def transactions(self, rows):
        return pd.DataFrame(rows, columns=[
            "msno", "transaction_date", "is_auto_renew", "is_cancel",
        ])

    def test_one_row_per_requested_customer_and_no_label_predictors(self):
        history = self.history([("synthetic-a", "20170228", 120)])
        history["is_churn"] = 1  # An extra input column must never become a predictor.
        events = self.transactions([("synthetic-a", "20170227", 1, 1)])
        events["is_churn"] = 1
        original_history, original_events = history.copy(), events.copy()
        features, quality = build_features(
            history, [events], ["synthetic-a", "synthetic-absent"], self.cutoff,
        )
        self.assertEqual(features.columns.tolist(),
                         ["customer_id", "prediction_date", *FEATURE_COLUMNS])
        self.assertFalse(features.duplicated(["customer_id", "prediction_date"]).any())
        self.assertEqual(len(features), 2)
        rows = features.set_index("customer_id")
        self.assertEqual(rows.loc["synthetic-a", "recorded_minutes_28d"], 2)
        self.assertEqual(rows.loc["synthetic-a", "cancellation_events_28d"], 1)
        self.assertEqual(rows.loc["synthetic-absent", "history_coverage"],
                         "No pre-cutoff source records")
        self.assertTrue(pd.isna(rows.loc["synthetic-absent", "recorded_minutes_28d"]))
        self.assertTrue(pd.isna(rows.loc["synthetic-absent", "days_since_last_transaction"]))
        self.assertEqual(quality["listening"]["customers_without_any_source_records"], 1)
        pd.testing.assert_frame_equal(history, original_history)
        pd.testing.assert_frame_equal(events, original_events)

    def test_cutoff_and_future_records_cannot_change_feature_values(self):
        history = self.history([
            ("synthetic-a", "20170201", 60),
            ("synthetic-a", "20170222", 60),
            ("synthetic-a", "20170228", 60),
        ])
        events = self.transactions([("synthetic-a", "20170228", 1, 0)])
        original, _ = build_features(history, [events], ["synthetic-a"], self.cutoff)
        changed, _ = build_features(
            pd.concat([history, self.history([
                ("synthetic-a", "20170301", 1e12),
                ("synthetic-a", "20170401", 1e12),
                ("not-in-cohort", "20170228", 1e12),
            ])]),
            [pd.concat([events, self.transactions([
                ("synthetic-a", "20170301", 0, 1),
                ("synthetic-a", "20170401", 0, 1),
            ])])], ["synthetic-a"], self.cutoff,
        )
        pd.testing.assert_frame_equal(original, changed)
        self.assertEqual(original.loc[0, "recorded_active_days_7d"], 2)
        self.assertEqual(original.loc[0, "recorded_active_days_28d"], 3)
        # Moving the prediction date is an explicit change, not a hidden constant.
        earlier, _ = build_features(history, [events], ["synthetic-a"], "2017-02-28")
        self.assertEqual(earlier.loc[0, "recorded_active_days_7d"], 1)
        self.assertEqual(earlier.loc[0, "latest_auto_renew_status"], "No earlier transaction")

    def test_duplicates_invalid_history_and_ambiguous_latest_flags_are_visible(self):
        history = self.history([
            ("synthetic-a", "20161001", 60),
            ("synthetic-a", "20170201", 60),
            ("synthetic-a", "20170201", 90),
            ("synthetic-a", "20170202", -1),
            ("synthetic-a", "20170203", "invalid"),
            ("synthetic-a", "bad-date", 60),
        ])
        events = self.transactions([
            ("synthetic-a", "20170228", 0, 0),
            ("synthetic-a", "20170228", 1, 1),
            ("synthetic-a", "bad-date", 1, 1),
        ])
        features, quality = build_features(history, [events], ["synthetic-a"], self.cutoff)
        self.assertEqual(features.loc[0, "latest_auto_renew_status"],
                         "Ambiguous latest-day flags")
        self.assertTrue(features.loc[0, "has_unusable_records_90d"])
        self.assertTrue(pd.isna(features.loc[0, "recent_to_baseline_ratio"]))
        self.assertEqual(quality["listening"]["rows_excluded_for_duplicate_day_keys"], 2)
        self.assertEqual(quality["listening"]["rows_with_invalid_duration"], 2)
        self.assertEqual(quality["transactions"]["invalid_date_rows"], 1)

    def test_personal_baseline_retains_undefined_ratio_for_unobserved_customers(self):
        history = self.history([
            ("synthetic-observed", day.strftime("%Y%m%d"),
             60 if day < pd.Timestamp("2017-02-01") else 30)
            for day in pd.date_range("2016-12-01", "2017-02-28")
        ])
        features, _ = build_features(history, [], ["synthetic-observed", "synthetic-absent"],
                                     self.cutoff)
        rows = features.set_index("customer_id")
        self.assertAlmostEqual(rows.loc["synthetic-observed", "recent_to_baseline_ratio"], 0.5)
        self.assertTrue(pd.isna(rows.loc["synthetic-absent", "recent_to_baseline_ratio"]))
        self.assertFalse(rows.loc["synthetic-absent", "baseline_comparison_eligible"])

    def test_invalid_customer_keys_dates_and_schema_are_rejected(self):
        empty = self.history([])
        for ids in [[], ["synthetic-a", "synthetic-a"], [None], [""]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_features(empty, [], ids, self.cutoff)
        for date in [pd.NaT, "2017-03-01 12:00", "2017-03-01T00:00:00Z"]:
            with self.subTest(date=date), self.assertRaises(ValueError):
                build_features(empty, [], ["synthetic-a"], date)
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame(), [], ["synthetic-a"], self.cutoff)
        with self.assertRaises(ValueError):
            build_features(empty, [pd.DataFrame()], ["synthetic-a"], self.cutoff)

    def test_csv_chunks_and_whole_input_produce_the_same_features(self):
        events = self.transactions([
            ("synthetic-a", "20170228", 0, 1),
            ("synthetic-a", "20170228", 1, 0),
            ("synthetic-absent", "20170301", 1, 1),
        ])
        history = self.history([])
        ids = ["synthetic-a", "synthetic-absent"]
        expected, quality = build_features(history, [events], ids, self.cutoff)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            events.assign(membership_expire_date="20170315").to_csv(path, index=False)
            preview = transaction_chunks(path, chunksize=1, all_columns=True)
            try:
                self.assertIn("membership_expire_date", next(preview).columns)
            finally:
                preview.close()
            actual, actual_quality = build_features(
                history, transaction_chunks(path, chunksize=1), ids, self.cutoff,
            )
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(quality, actual_quality)

    def test_invalid_flags_are_counted_and_never_treated_as_known_renewal(self):
        events = self.transactions([("synthetic-a", "20170228", 2, -1)])
        features, quality = build_features(self.history([]), [events], ["synthetic-a"],
                                           self.cutoff)
        self.assertEqual(features.loc[0, "latest_auto_renew_status"],
                         "Ambiguous latest-day flags")
        self.assertEqual(features.loc[0, "cancellation_events_28d"], 0)
        self.assertEqual(quality["transactions"]["invalid_auto_renew_rows"], 1)
        self.assertEqual(quality["transactions"]["invalid_cancel_rows"], 1)

    def test_date_audit_scans_all_rows_and_refreshes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cache = Path(directory) / "events.csv", Path(directory) / "ranges.json"
            events = pd.DataFrame({"date": ["20151231", "20170301", "20160230", None]})
            events.to_csv(path, index=False)
            ranges = source_date_ranges(path, ["date"], cache)
            self.assertEqual(ranges[0]["earliest"], "2015-12-31")
            self.assertEqual(ranges[0]["latest"], "2017-03-01")
            self.assertEqual(ranges[0]["rows_checked"], 4)
            self.assertEqual(ranges[0]["invalid_dates"], 2)
            with patch("churn_retention.features.csv_chunks", side_effect=AssertionError):
                self.assertEqual(source_date_ranges(path, ["date"], cache), ranges)
            pd.concat([events, pd.DataFrame({"date": ["20361015"]})]).to_csv(path, index=False)
            self.assertEqual(source_date_ranges(path, ["date"], cache)[0]["latest"], "2036-10-15")

    def test_date_free_labels_are_explicit_and_schema_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path, cache = Path(directory) / "labels.csv", Path(directory) / "ranges.json"
            pd.DataFrame({"msno": ["synthetic-a"], "is_churn": [1]}).to_csv(path, index=False)
            result = source_date_ranges(path, [], cache)[0]
            self.assertIsNone(result["earliest"])
            self.assertIsNone(result["latest"])
            self.assertEqual(result["scope"], "Label file: no date columns")
            pd.DataFrame({"date": ["20170301"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                source_date_ranges(path, [], cache)


if __name__ == "__main__":
    unittest.main()
