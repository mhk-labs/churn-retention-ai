"""Synthetic checks that the provider's churn-labelling rules are reproduced."""

import unittest

try:
    import pandas as pd
except ImportError:
    pd = None
else:
    from churn_retention.labels import LABEL_COLUMNS, provider_labels


def events(rows):
    """Rows: (id, transaction_date, expiry, is_cancel); a 30-day plan by default."""
    return pd.DataFrame(
        [(msno, 41, 30, 149, date, expiry, cancel) for msno, date, expiry, cancel in rows],
        columns=["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                 "transaction_date", "membership_expire_date", "is_cancel"])


# Provider example: January history, February expiries, data through 31 March.
WINDOW = {"history_start": 20170101, "history_end": 20170131, "expiry_start": 20170201,
          "expiry_end": 20170228, "data_end": 20170331}


@unittest.skipIf(pd is None, "Install pandas to check label construction.")
class ProviderLabelTests(unittest.TestCase):
    def label(self, rows, **overrides):
        return provider_labels(events(rows), **{**WINDOW, **overrides})

    def test_renewal_gap_rules(self):
        labels = self.label([
            ("synthetic-renews", 20170110, 20170210, 0), ("synthetic-renews", 20170215, 20170315, 0),
            ("synthetic-late", 20170105, 20170205, 0), ("synthetic-late", 20170307, 20170407, 0),
            ("synthetic-silent", 20170101, 20170201, 0),
        ])
        self.assertEqual(list(labels.columns), list(LABEL_COLUMNS))
        self.assertEqual(labels.loc["synthetic-renews", "renewal_gap_days"], 5)
        self.assertEqual(labels.loc["synthetic-renews", "is_churn"], 0)
        self.assertEqual(labels.loc["synthetic-late", "renewal_gap_days"], 30)  # 30 days = churn
        self.assertEqual(labels.loc["synthetic-late", "is_churn"], 1)
        self.assertTrue(pd.isna(labels.loc["synthetic-silent", "renewal_date"]))
        self.assertEqual(labels.loc["synthetic-silent", "is_churn"], 1)
        self.assertTrue(labels["outcome_window_complete"].all())

    def test_only_customers_expiring_in_window_are_candidates(self):
        labels = self.label([
            ("synthetic-march", 20170120, 20170320, 0),       # expires after the window
            ("synthetic-december", 20161201, 20170210, 0),   # no event in history window
        ])
        self.assertTrue(labels.empty)

    def test_same_day_cancel_follows_subscription(self):
        # Same plan on one day: the renewal is ordered first, so the cancellation's
        # earlier expiry is the last one seen.
        labels = self.label([
            ("synthetic-a", 20170115, 20170220, 1), ("synthetic-a", 20170115, 20170315, 0),
        ])
        self.assertEqual(labels.loc["synthetic-a", "last_expire"], 20170220)

    def test_later_cancellation_moves_expiry_earlier(self):
        labels = self.label([
            ("synthetic-a", 20170110, 20170228, 0),
            ("synthetic-a", 20170205, 20170206, 1),           # cancels; access ends 6 Feb
            ("synthetic-a", 20170310, 20170410, 0),           # 32 days after 6 Feb
        ])
        self.assertEqual(labels.loc["synthetic-a", "renewal_gap_days"], 32)
        self.assertEqual(labels.loc["synthetic-a", "is_churn"], 1)

    def test_churn_after_data_end_is_marked_immature(self):
        labels = self.label([("synthetic-a", 20170125, 20170225, 0)], data_end=20170310)
        self.assertEqual(labels.loc["synthetic-a", "is_churn"], 1)
        self.assertFalse(labels.loc["synthetic-a", "outcome_window_complete"])

    def test_events_after_data_end_are_ignored(self):
        labels = self.label([
            ("synthetic-a", 20170125, 20170225, 0), ("synthetic-a", 20170305, 20170405, 0),
        ], data_end=20170301)
        self.assertEqual(labels.loc["synthetic-a", "is_churn"], 1)
        self.assertFalse(labels.loc["synthetic-a", "outcome_window_complete"])


if __name__ == "__main__":
    unittest.main()
