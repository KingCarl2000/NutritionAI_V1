import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import optuna
from optuna.exceptions import TrialPruned

from utils.ml_utils.model.model_factory import BaseModelWrapper, ModelFactory


class ModelFactoryTests(unittest.TestCase):
    def test_trial_factory_fits_and_round_trips_ridge_wrapper(self):
        trial = optuna.trial.FixedTrial({"ridge.alpha": 1.0})
        model, model_name = ModelFactory.create_model_from_trial(
            trial,
            model_name="ridge",
        )
        features = np.array([[0.0], [1.0], [2.0], [3.0]])
        targets = np.array([0.0, 1.0, 2.0, 3.0])
        model.fit(features, targets)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nested" / "model.joblib"
            model.save(model_path)
            loaded = BaseModelWrapper.load(model_path)

        self.assertEqual(model_name, "ridge")
        self.assertIsInstance(loaded, BaseModelWrapper)
        np.testing.assert_allclose(
            loaded.predict(features),
            model.predict(features),
        )

    def test_xgboost_wrapper_prunes_using_validation_metric(self):
        model = ModelFactory.create_model(
            "xgboost",
            {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        )
        trial = Mock()
        trial.should_prune.return_value = True
        features = np.arange(20, dtype=float).reshape(-1, 1)
        targets = np.arange(20, dtype=float)

        with self.assertRaises(TrialPruned):
            model.fit(
                features[:15],
                targets[:15],
                X_val=features[15:],
                y_val=targets[15:],
                trial=trial,
            )

        trial.report.assert_called_once()
        self.assertEqual(model.estimator.get_params()["callbacks"], [])

    def test_fit_requires_validation_features_and_targets_together(self):
        model = ModelFactory.create_model("ridge")

        with self.assertRaisesRegex(ValueError, "provided together"):
            model.fit(np.array([[0.0]]), np.array([0.0]), X_val=np.array([[1.0]]))

        xgboost_model = ModelFactory.create_model(
            "xgboost",
            {"n_estimators": 1, "n_jobs": 1},
        )
        with self.assertRaisesRegex(ValueError, "required to prune"):
            xgboost_model.fit(
                np.array([[0.0], [1.0]]),
                np.array([0.0, 1.0]),
                trial=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
