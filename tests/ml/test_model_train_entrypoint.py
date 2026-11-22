import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import joblib
import mlflow
import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from src.pipelines.ml_pipelines.calories_pred.model_train import run_training


class ModelTrainEntrypointTests(unittest.TestCase):
    def test_run_training_splits_target_and_writes_model_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            previous_tracking_uri = mlflow.get_tracking_uri()
            self.addCleanup(mlflow.set_tracking_uri, previous_tracking_uri)
            tracking_uri = "sqlite:///:memory:"
            experiment_name = f"entrypoint-test-{uuid4().hex}"
            train_path = root / "train.npy"
            val_path = root / "val.npy"
            preprocessor_path = root / "preprocessor.pkl"
            model_path = root / "models" / "best_model.pkl"
            report_path = root / "reports" / "training_report.json"

            train_features = np.arange(20, dtype=float).reshape(-1, 1)
            train_targets = 3 * train_features[:, 0] + 2
            val_features = np.array([[20.0], [21.0], [22.0]])
            val_targets = 3 * val_features[:, 0] + 2
            np.save(train_path, np.column_stack((train_features, train_targets)))
            np.save(val_path, np.column_stack((val_features, val_targets)))
            joblib.dump(StandardScaler().fit(train_features), preprocessor_path)

            cfg = OmegaConf.create(
                {
                    "train_params": {
                        "artifacts": {
                            "transformed_train_path": str(train_path),
                            "transformed_val_path": str(val_path),
                            "preprocessor_path": str(preprocessor_path),
                            "best_model_path": str(model_path),
                            "training_report_path": str(report_path),
                        }
                    },
                    "training": {
                        "experiment_name": experiment_name,
                        "tracking_uri": tracking_uri,
                        "direction": "auto",
                        "metric_name": "auto",
                        "metric_names": [
                            "mean_squared_error",
                            "root_mean_squared_log_error",
                            "r2_score",
                        ],
                        "n_trials": 1,
                        "n_jobs": 1,
                        "model_name": "ridge",
                        "seed": 42,
                    },
                }
            )

            report = run_training(cfg)

            for metric_name in cfg.training.metric_names:
                self.assertTrue(
                    model_path.with_name(
                        f"{model_path.stem}_{metric_name}{model_path.suffix}"
                    ).is_file()
                )
            self.assertEqual(report["data"]["feature_count"], 1)
            self.assertEqual(report["data"]["train_rows"], 20)
            self.assertEqual(report["data"]["validation_rows"], 3)
            self.assertEqual(len(report["metric_results"]), 3)
            self.assertEqual(
                report["metric_results"]["r2_score"]["model_type"], "ridge"
            )
            self.assertEqual(json.loads(report_path.read_text())["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
