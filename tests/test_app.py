"""Exercise the synthetic demo through Streamlit without starting a server."""

import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Install the app extra to check the Streamlit demo.")
class AppTests(unittest.TestCase):
    def test_customer_selection_displays_the_matching_synthetic_risk(self):
        path = Path(__file__).resolve().parents[1] / "churn_retention" / "app.py"
        app = AppTest.from_file(str(path)).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "72%")
        app.selectbox[0].select("demo-002").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "18%")


if __name__ == "__main__":
    unittest.main()
