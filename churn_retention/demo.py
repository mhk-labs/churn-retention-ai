"""Offline synthetic fixture; no trained model, SHAP computation, or API calls."""

import json
from copy import deepcopy

_CUSTOMERS = [
    {
        "customer_id": "demo-001",
        "prediction_date": "2017-02-01",
        "churn_probability": 0.72,
        "model_version": "synthetic-fixture-v1",
        "is_synthetic": True,
        "drivers": [
            {
                "feature": "listening_days_30d",
                "value": 3,
                "direction": "increases_risk",
                "description": "Illustrative lower activity: 3 days versus 12 previously.",
            }
        ],
    },
    {
        "customer_id": "demo-002",
        "prediction_date": "2017-02-01",
        "churn_probability": 0.18,
        "model_version": "synthetic-fixture-v1",
        "is_synthetic": True,
        "drivers": [
            {
                "feature": "listening_days_30d",
                "value": 24,
                "direction": "decreases_risk",
                "description": "Illustrative regular listening activity.",
            }
        ],
    },
]


def load_demo_customers() -> list[dict]:
    """Return independent fictional records for development."""
    return deepcopy(_CUSTOMERS)


def main() -> None:
    """Print the shared mock handoff without data or credentials."""
    print(json.dumps({"mode": "synthetic_demo", "customers": load_demo_customers()}, indent=2))


if __name__ == "__main__":
    main()
