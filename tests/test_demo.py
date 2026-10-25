"""Check the synthetic handoff used by app and pipeline developers."""

import unittest
from datetime import date

from churn_retention.demo import load_demo_customers


class DemoContractTests(unittest.TestCase):
    def test_records_have_valid_unique_keys_and_probabilities(self):
        rows = load_demo_customers()
        self.assertTrue(rows)
        keys = set()
        for row in rows:
            key = (row["customer_id"], row["prediction_date"])
            self.assertNotIn(key, keys)
            keys.add(key)
            date.fromisoformat(row["prediction_date"])
            self.assertTrue(row["is_synthetic"])
            self.assertTrue(row["model_version"].startswith("synthetic-"))
            self.assertGreaterEqual(row["churn_probability"], 0)
            self.assertLessEqual(row["churn_probability"], 1)
            for driver in row["drivers"]:
                self.assertIn(driver["direction"], ("increases_risk", "decreases_risk"))
                self.assertTrue(driver["feature"])
                self.assertTrue(driver["description"])

    def test_callers_cannot_mutate_shared_fixture(self):
        first = load_demo_customers()
        first[0]["drivers"][0]["value"] = -999
        self.assertNotEqual(load_demo_customers()[0]["drivers"][0]["value"], -999)


if __name__ == "__main__":
    unittest.main()
