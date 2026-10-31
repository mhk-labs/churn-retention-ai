"""Keep real data and executed outputs out of the Kaggle upload package."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class KaggleStagingTests(unittest.TestCase):
    def test_only_clean_notebook_and_private_metadata_are_staged(self):
        script = Path(__file__).resolve().parents[1] / "churn_retention" / "kaggle.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "churn_retention").mkdir()
            (root / "notebooks").mkdir()
            (root / "data" / "raw").mkdir(parents=True)
            shutil.copyfile(script, root / "churn_retention" / script.name)
            (root / "data" / "raw" / "customers.csv").write_text("customer_id\ndemo-001\n")
            (root / ".env").write_text("SYNTHETIC_SECRET=fixture-only\n")
            fixture = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {"widgets": {"synthetic_state": "demo-001"}},
                "cells": [{
                    "id": "synthetic-cell", "cell_type": "code", "metadata": {},
                    "source": ["print('synthetic example')"], "execution_count": 1,
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": "demo-001"}],
                }],
            }
            (root / "notebooks" / "01_eda.ipynb").write_text(json.dumps(fixture))
            subprocess.run(
                [sys.executable, str(root / "churn_retention" / script.name), "--owner", "demo-user"],
                check=True, capture_output=True, text=True,
            )
            output = root / "data" / "processed" / "kaggle-eda"
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"01_eda.ipynb", "kernel-metadata.json"}
            )
            notebook = json.loads((output / "01_eda.ipynb").read_text())
            self.assertEqual(notebook["cells"][0]["outputs"], [])
            self.assertIsNone(notebook["cells"][0]["execution_count"])
            self.assertNotIn("widgets", notebook["metadata"])
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["enable_gpu"], "false")
            self.assertEqual(metadata["competition_sources"], ["kkbox-churn-prediction-challenge"])
            self.assertEqual(metadata["dataset_sources"], [])


if __name__ == "__main__":
    unittest.main()
