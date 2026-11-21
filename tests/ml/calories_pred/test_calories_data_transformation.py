import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nutrition_core.exception.exception import DataTransformationError
from src.pipelines.ml_pipelines.calories_pred.data_trans import (
    transform_splits,
    verify_gatekeeper_reports,
)


class FitTrackingPreprocessor:
    def __init__(self):
        self.calls = []

    def fit_transform(self, features):
        self.calls.append(("fit_transform", features["feature"].tolist()))
        return features[["feature"]].to_numpy()

    def transform(self, features):
        self.calls.append(("transform", features["feature"].tolist()))
        return features[["feature"]].to_numpy()


class CaloriesTransformationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def write_gatekeeper_reports(self, validation_status="PASSED", drift_share=0.0):
        validation_path = self.temp_path / "validation_report.json"
        drift_path = self.temp_path / "drift_report.json"
        validation_path.write_text(
            json.dumps(
                {
                    "status": validation_status,
                    **{
                        split: {"status": "PASSED", "error_count": 0}
                        for split in ("train", "val", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        drift_path.write_text(
            json.dumps(
                {
                    "metrics": [
                        {
                            "metric_name": "DriftedColumnsCount(...)",
                            "config": {"drift_share": 0.5},
                            "value": {"share": drift_share},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return validation_path, drift_path

    def test_transform_splits_fits_only_training_features(self):
        preprocessor = FitTrackingPreprocessor()
        feature_splits = {
            "train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "val": pd.DataFrame({"feature": [100.0]}),
            "test": pd.DataFrame({"feature": [200.0]}),
        }
        target_splits = {
            "train": pd.Series([10.0, 20.0]),
            "val": pd.Series([30.0]),
            "test": pd.Series([40.0]),
        }

        transformed = transform_splits(preprocessor, feature_splits, target_splits)

        self.assertEqual(
            preprocessor.calls,
            [
                ("fit_transform", [1.0, 2.0]),
                ("transform", [100.0]),
                ("transform", [200.0]),
            ],
        )
        np.testing.assert_array_equal(
            transformed["val"], np.array([[100.0, 30.0]])
        )
        np.testing.assert_array_equal(
            transformed["test"], np.array([[200.0, 40.0]])
        )

    def test_gatekeeper_reports_must_pass_validation_and_drift(self):
        validation_path, drift_path = self.write_gatekeeper_reports()

        status = verify_gatekeeper_reports(validation_path, drift_path)

        self.assertEqual(
            status,
            {
                "validation_status": "PASSED",
                "drift_status": "PASSED",
                "drift_share": 0.0,
            },
        )

    def test_gatekeeper_blocks_failed_validation_or_drift(self):
        for validation_status, drift_share in (
            ("FAILED", 0.0),
            ("PASSED", 0.5),
        ):
            with self.subTest(
                validation_status=validation_status, drift_share=drift_share
            ):
                validation_path, drift_path = self.write_gatekeeper_reports(
                    validation_status, drift_share
                )
                with self.assertRaises(DataTransformationError):
                    verify_gatekeeper_reports(validation_path, drift_path)

