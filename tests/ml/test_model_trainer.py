import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from utils.ml_utils.model.model_trainer import ModelTrainer


class ModelTrainerTests(unittest.TestCase):
    def test_optimize_tracks_parent_children_and_best_model_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_tracking_uri = mlflow.get_tracking_uri()
            self.addCleanup(mlflow.set_tracking_uri, previous_tracking_uri)
            root = Path(temp_dir)
            tracking_uri = "sqlite:///:memory:"
            experiment_name = f"model-trainer-test-{uuid4().hex}"
            mlflow.set_tracking_uri(tracking_uri)
            MlflowClient(tracking_uri=tracking_uri).create_experiment(
                experiment_name,
                artifact_location=(root / "artifacts").as_uri(),
            )
            local_model_path = root / "export" / "best_model.pkl"
            trainer = ModelTrainer(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                seed=7,
            )
            X_train = np.arange(20, dtype=float).reshape(-1, 1)
            y_train = 2 * X_train[:, 0] + 1
            X_val = np.array([[20.0], [21.0], [22.0]])
            y_val = 2 * X_val[:, 0] + 1

            wrapper, best_params, best_score = trainer.optimize_and_train(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=2,
                model_name="ridge",
                model_output_path=local_model_path,
            )

            self.assertTrue(local_model_path.exists())
            self.assertEqual(wrapper.model_name, "ridge")
            self.assertEqual(best_params["model_type"], "ridge")
            self.assertGreater(best_score, 0.0)
            self.assertLessEqual(best_score, 1.0)

            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.assertIsNotNone(experiment)
            runs = MlflowClient(tracking_uri=tracking_uri).search_runs(
                [experiment.experiment_id]
            )
            parent_runs = [
                run for run in runs if run.data.tags.get("run_type") == "hpo_parent"
            ]
            self.assertEqual(len(parent_runs), 1)
            parent = parent_runs[0]
            child_runs = [
                run
                for run in runs
                if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
            ]
            self.assertEqual(len(child_runs), 2)
            self.assertEqual(
                sum(run.data.tags.get("is_best_run") == "true" for run in child_runs),
                1,
            )
            self.assertTrue(
                any(
                    artifact.path.endswith("best_model.pkl")
                    for artifact in MlflowClient(
                        tracking_uri=tracking_uri
                    ).list_artifacts(parent.info.run_id, "best_model")
                )
            )

    def test_save_best_model_requires_completed_training(self):
        trainer = ModelTrainer(experiment_name="model-trainer-test")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "optimization first"):
                trainer.save_best_model(Path(temp_dir) / "model.pkl")


if __name__ == "__main__":
    unittest.main()
