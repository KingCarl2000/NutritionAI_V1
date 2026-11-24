import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.pipelines.ml_pipelines.calories_pred.model_register import (
    register_models_from_report,
)


class ModelRegisterTests(unittest.TestCase):
    def test_imports_each_model_using_uri_from_training_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "training_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "metric_results": {
                            "mean_squared_error": {
                                "mlflow_model_uri": "models:/mse-model"
                            },
                            "r2_score": {"mlflow_model_uri": "models:/r2-model"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            import_model = Mock(side_effect=["bento-mse", "bento-r2"])
            bentoml = ModuleType("bentoml")
            bentoml.mlflow = SimpleNamespace(import_model=import_model)

            with patch.dict(sys.modules, {"bentoml": bentoml}):
                result = register_models_from_report(report_path)

            self.assertEqual(
                result,
                {
                    "mean_squared_error": "bento-mse",
                    "r2_score": "bento-r2",
                },
            )
            self.assertEqual(
                [call.kwargs["model_uri"] for call in import_model.call_args_list],
                ["models:/mse-model", "models:/r2-model"],
            )
            self.assertEqual(
                [call.kwargs["name"] for call in import_model.call_args_list],
                [
                    "calories_predictor_mean_squared_error",
                    "calories_predictor_r2_score",
                ],
            )

    def test_requires_logged_model_uri_for_each_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "training_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "metric_results": {"r2_score": {}},
                    }
                ),
                encoding="utf-8",
            )
            bentoml = ModuleType("bentoml")
            bentoml.mlflow = SimpleNamespace(import_model=Mock())

            with (
                patch.dict(sys.modules, {"bentoml": bentoml}),
                self.assertRaisesRegex(ValueError, "model URI"),
            ):
                register_models_from_report(report_path)


if __name__ == "__main__":
    unittest.main()
