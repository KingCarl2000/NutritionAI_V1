import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import yaml

from src.nutrition_core.exception.exception import DataTransformationError
from src.pipelines.ml_pipelines.calories_pred.data_trans import (
    PROJECT_ROOT,
    CaloriesDataTransformation,
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

    def test_load_datasets_uses_dvc_for_repo_artifacts(self):
        config_path = self.temp_path / "data_transformation.yaml"
        config = {
            "artifacts": {
                f"valid_{split}_path": (
                    f"Artifacts/data/ml_calories/validated/{split}.csv"
                )
                for split in ("train", "val", "test")
            },
            "column_mapping": {
                "numerical_features": ["Age"],
                "categorical_features": ["Gender"],
                "target": "Calories",
                "id_column": "User_ID",
            },
            "preprocessing": {},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        dataset = pd.DataFrame(
            {
                "Age": [25],
                "Gender": ["F"],
                "Calories": [100],
                "User_ID": [1],
            }
        )
        dvc_handler = Mock()
        dvc_handler.get_dataset.side_effect = [dataset.copy() for _ in range(3)]

        with patch(
            "src.pipelines.ml_pipelines.calories_pred.data_trans.DVCHandler",
            return_value=dvc_handler,
        ) as handler_class:
            transformation = CaloriesDataTransformation(config_path)
            datasets, missing_values = transformation._load_datasets()

        handler_class.assert_called_once_with(repo=str(PROJECT_ROOT))
        self.assertEqual(
            dvc_handler.get_dataset.call_args_list,
            [
                call(
                    data_path=f"Artifacts/data/ml_calories/validated/{split}.csv",
                    format_type="csv",
                    fallback_path=str(
                        PROJECT_ROOT
                        / "Artifacts"
                        / "data"
                        / "ml_calories"
                        / "validated"
                        / f"{split}.csv"
                    ),
                )
                for split in ("train", "val", "test")
            ],
        )
        self.assertEqual(set(datasets), {"train", "val", "test"})
        self.assertEqual(missing_values, {"train": 0, "val": 0, "test": 0})

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

