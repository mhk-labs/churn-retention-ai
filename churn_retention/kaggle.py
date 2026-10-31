"""Prepare notebook 01 for a private Kaggle upload; do not upload data."""

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Your Kaggle username")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.owner):
        parser.error("Use your Kaggle username slug, containing lowercase letters/numbers/_/-.")

    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "processed" / "kaggle-eda"
    output.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((root / "notebooks" / "01_eda.ipynb").read_text())
    notebook["metadata"].pop("widgets", None)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    (output / "01_eda.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.owner}/churn-retention-ai-eda",
        "title": "Churn Retention AI EDA",
        "code_file": "01_eda.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": ["kkbox-churn-prediction-challenge"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Staged notebook and metadata only: {output}")
    print("Private CPU notebook; official competition input attached; no local data copied.")
    print("This command did not upload or run anything.")


if __name__ == "__main__":
    main()
