import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
            storage_url = (
                f"sqlite:///file:calories_{uuid4().hex}"
                "?mode=memory&cache=shared&uri=true"
            )
            tracking_uri = (
                f"sqlite:///file:mlflow_{uuid4().hex}"
                "?mode=memory&cache=shared&uri=true"
            )

            train_raw = pd.DataFrame(
                {"Age": np.arange(20, dtype=float), "Gender": ["F", "M"] * 10}
            )
            val_raw = pd.DataFrame(
                {"Age": [20.0, 21.0, 22.0], "Gender": ["X", "F", "M"]}
            )
            preprocessor = ColumnTransformer(
                [
                    (
                        "numerical",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        ["Age"],
                    ),
                    (
                        "categorical",
                        Pipeline(
                            [
                                (
                                    "imputer",
                                    SimpleImputer(strategy="most_frequent"),
                                ),
                                (
                                    "encoder",
                                    OneHotEncoder(
                                        handle_unknown="ignore",
                                        sparse_output=False,
                                    ),
                                ),
                            ]
                        ),
                        ["Gender"],
                    ),
                ],
                sparse_threshold=0.0,
            )
            train_features = preprocessor.fit_transform(train_raw)
            val_features = preprocessor.transform(val_raw)
            train_targets = 3 * train_raw["Age"].to_numpy() + 2
            val_targets = 3 * val_raw["Age"].to_numpy() + 2
            np.save(train_path, np.column_stack((train_features, train_targets)))
            np.save(val_path, np.column_stack((val_features, val_targets)))
            joblib.dump(preprocessor, preprocessor_path)

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
                    "optuna": {
                        "study_name": "calories_regression",
                        "storage": storage_url,
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
                self.assertTrue(
                    model_path.with_name(
                        f"{model_path.stem}_{metric_name}_pipeline{model_path.suffix}"
                    ).is_file()
                )
            self.assertEqual(report["data"]["feature_count"], 3)
            self.assertEqual(report["data"]["train_rows"], 20)
            self.assertEqual(report["data"]["validation_rows"], 3)
            self.assertEqual(len(report["metric_results"]), 3)
            self.assertEqual(
                report["metric_results"]["r2_score"]["model_type"], "ridge"
            )
            saved_report = json.loads(report_path.read_text())
            self.assertEqual(saved_report["status"], "SUCCESS")
            self.assertEqual(saved_report["run_id"], report["run_id"])
            mlflow_run = mlflow.get_run(report["run_id"])
            self.assertEqual(mlflow_run.info.run_id, report["run_id"])

            model_uri = report["metric_results"]["r2_score"]["mlflow_model_uri"]
            local_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
            )
            self.assertTrue((Path(local_model_path) / "MLmodel").is_file())
            self.assertTrue((Path(local_model_path) / "requirements.txt").is_file())
            self.assertTrue((Path(local_model_path) / "input_example.json").is_file())
            logged_model = mlflow.models.Model.load(
                str(Path(local_model_path) / "MLmodel")
            )
            self.assertIsNotNone(logged_model.signature)
            self.assertIsNotNone(logged_model.signature.inputs)
            self.assertIsNotNone(logged_model.signature.outputs)
            loaded_model = mlflow.sklearn.load_model(model_uri)
            raw_prediction = loaded_model.predict(
                pd.DataFrame({"Age": [0.0], "Gender": ["F"]})
            )
            self.assertEqual(raw_prediction.shape, (1,))
            inference_pipeline_path = Path(
                report["metric_results"]["r2_score"]["inference_pipeline_path"]
            )
            pipeline = joblib.load(inference_pipeline_path)
            self.assertEqual(
                pipeline.predict(pd.DataFrame({"Age": [0.0], "Gender": ["F"]})).shape,
                (1,),
            )
            study_storage = None
            try:
                for metric_name in cfg.training.metric_names:
                    study = optuna.load_study(
                        study_name=f"calories_regression_{metric_name}",
                        storage=storage_url,
                    )
                    self.assertEqual(len(study.trials), 1)
                    study_storage = study._storage
            finally:
                if study_storage is not None:
                    study_storage.remove_session()
                    study_storage._backend.engine.dispose()


if __name__ == "__main__":
    unittest.main()
