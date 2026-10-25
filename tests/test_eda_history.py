"""Offline synthetic checks for the EDA notebook's temporal comparison rules."""

import ast
import json
import unittest
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


@unittest.skipIf(pd is None, "Install the notebooks extra to check pandas EDA summaries.")
class EDAHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only pure summary functions: no notebook setup, downloads or data reads.
        path = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
        notebook = json.loads(path.read_text())
        cls.functions = {"pd": pd}
        wanted = {"listening_windows", "renewal_by_customer"}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            nodes = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in wanted]
            if nodes:
                # Execute trusted repository functions, never source data or setup cells.
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),  # noqa: S102
                     cls.functions)
        assert wanted.issubset(cls.functions)
        cls.cutoff = pd.Timestamp("2017-03-01")

    def listening(self, rows, ids):
        frame = pd.DataFrame(rows, columns=["msno", "date", "total_secs"])
        return self.functions["listening_windows"](frame, ids, self.cutoff)

    def test_cutoff_and_window_boundaries_ignore_future_changes(self):
        rows = [
            ("synthetic-a", "20161130", 60),
            ("synthetic-a", "20161201", 60),
            ("synthetic-a", "20170201", 60),
            ("synthetic-a", "20170222", 60),
            ("synthetic-a", "20170228", 60),
        ]
        before, _ = self.listening(rows, {"synthetic-a"})
        after, quality = self.listening(rows + [
            ("synthetic-a", "20170301", 1e12),
            ("synthetic-a", "20170320", 1e12),
            ("outside-sample", "20170228", 1e12),
        ], {"synthetic-a"})
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(quality["rows_on_or_after_cutoff_excluded"], 2)
        expected = {"7d": 2, "28d": 3, "90d": 4, "all": 5}
        for suffix, count in expected.items():
            self.assertEqual(after.loc["synthetic-a", f"recorded_active_days_{suffix}"], count)
        self.assertNotIn("is_churn", after.columns)

    def test_personal_baseline_normalises_period_lengths_and_preserves_unknowns(self):
        # Exactly one minute per calendar day in each period: the ratio must be 1.
        rows = []
        for customer, recent_seconds in [("synthetic-steady", 60),
                                          ("synthetic-falling", 30),
                                          ("synthetic-growing", 120)]:
            for day in pd.date_range("2016-12-01", "2017-02-28"):
                seconds = 60 if day < pd.Timestamp("2017-02-01") else recent_seconds
                rows.append((customer, day.strftime("%Y%m%d"), seconds))
        rows += [
            ("synthetic-older", "20161001", 60),
            ("synthetic-zero-baseline", "20161101", 60),
            ("synthetic-zero-baseline", "20170228", 60),
            ("synthetic-new", "20170228", 60),
        ]
        ids = {"synthetic-steady", "synthetic-falling", "synthetic-growing",
               "synthetic-older", "synthetic-zero-baseline",
               "synthetic-new", "synthetic-absent"}
        result, _ = self.listening(rows, ids)
        self.assertAlmostEqual(result.loc["synthetic-steady", "recent_to_baseline_ratio"], 1)
        self.assertAlmostEqual(result.loc["synthetic-falling", "recent_to_baseline_ratio"], 0.5)
        self.assertAlmostEqual(result.loc["synthetic-growing", "recent_to_baseline_ratio"], 2)
        self.assertEqual(result.loc["synthetic-older", "history_coverage"],
                         "Older records only; none in last 28 days")
        self.assertEqual(result.loc["synthetic-absent", "history_coverage"],
                         "No pre-cutoff source records")
        for customer in ["synthetic-zero-baseline", "synthetic-new", "synthetic-absent"]:
            self.assertTrue(pd.isna(result.loc[customer, "recent_to_baseline_ratio"]))
        self.assertTrue(pd.isna(result.loc["synthetic-absent", "recorded_minutes_all"]))

    def test_unusable_rows_are_reported_and_excluded_from_baseline_comparison(self):
        rows = [
            ("synthetic-a", "20161101", 60),
            ("synthetic-a", "20170201", 60),
            ("synthetic-a", "20170201", 120),
            ("synthetic-a", "20170202", -5),
            ("synthetic-a", "20170203", float("inf")),
            ("synthetic-a", "20170204", 0),
            ("synthetic-a", "20160230", 999),
        ]
        result, quality = self.listening(rows, {"synthetic-a"})
        self.assertEqual(quality["rows_excluded_for_duplicate_day_keys"], 2)
        self.assertEqual(quality["rows_with_invalid_duration"], 2)
        self.assertEqual(quality["invalid_date_rows"], 1)
        self.assertEqual(result.loc["synthetic-a", "recorded_active_days_28d"], 0)
        self.assertEqual(result.loc["synthetic-a", "recorded_minutes_28d"], 0)
        self.assertFalse(result.loc["synthetic-a", "baseline_comparison_eligible"])
        self.assertTrue(pd.isna(result.loc["synthetic-a", "recent_to_baseline_ratio"]))

    def test_subscription_events_handle_ties_and_chunks_without_future_information(self):
        events = pd.DataFrame([
            ("synthetic-a", "20170228", 0, 1),
            ("synthetic-a", "20170228", 1, 0),
            ("synthetic-a", "20170301", 1, 1),
            ("synthetic-b", "20170101", 1, 0),
        ], columns=["msno", "transaction_date", "is_auto_renew", "is_cancel"])
        ids = {"synthetic-a", "synthetic-b", "synthetic-absent"}
        summarize = self.functions["renewal_by_customer"]
        whole = summarize([events], ids, self.cutoff)
        chunked = summarize([events.iloc[:1], events.iloc[1:]], ids, self.cutoff)
        pd.testing.assert_frame_equal(whole, chunked)
        self.assertEqual(whole.loc["synthetic-a", "latest_auto_renew_status"],
                         "Ambiguous latest-day flags")
        self.assertEqual(whole.loc["synthetic-a", "transaction_events_all"], 2)
        self.assertEqual(whole.loc["synthetic-a", "cancellation_events_28d"], 1)
        self.assertEqual(whole.loc["synthetic-b", "transaction_events_28d"], 0)
        self.assertEqual(whole.loc["synthetic-b", "transaction_events_90d"], 1)
        self.assertEqual(whole.loc["synthetic-absent", "latest_auto_renew_status"],
                         "No earlier transaction")


if __name__ == "__main__":
    unittest.main()
