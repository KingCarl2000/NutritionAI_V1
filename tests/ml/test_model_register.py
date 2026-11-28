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
    def test_registers_each_saved_pipeline_in_bentoml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "training_report.json"
            mse_pipeline = root / "mse_pipeline.pkl"
            r2_pipeline = root / "r2_pipeline.pkl"
            mse_pipeline.touch()
            r2_pipeline.touch()
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-123",
                        "metric_results": {
                            "mean_squared_error": {
                                "inference_pipeline_path": str(mse_pipeline)
                            },
                            "r2_score": {"inference_pipeline_path": str(r2_pipeline)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            save_model = Mock(side_effect=["bento-mse", "bento-r2"])
            bentoml = ModuleType("bentoml")
            bentoml.sklearn = SimpleNamespace(save_model=save_model)

            with (
                patch.dict(sys.modules, {"bentoml": bentoml}),
                patch(
                    "src.pipelines.ml_pipelines.calories_pred.model_register.joblib.load",
                    side_effect=["mse-pipeline", "r2-pipeline"],
                ) as load_model,
            ):
                result = register_models_from_report(report_path)

            self.assertEqual(
                result,
                {
                    "mean_squared_error": "bento-mse",
                    "r2_score": "bento-r2",
                },
            )
            self.assertEqual(
                [call.args[0] for call in load_model.call_args_list],
                [mse_pipeline, r2_pipeline],
            )
            self.assertEqual(
                [call.kwargs["name"] for call in save_model.call_args_list],
                [
                    "calories_predictor_mean_squared_error",
                    "calories_predictor_r2_score",
                ],
            )

    def test_requires_saved_pipeline_for_each_metric(self):
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
            bentoml.sklearn = SimpleNamespace(save_model=Mock())

            with (
                patch.dict(sys.modules, {"bentoml": bentoml}),
                self.assertRaisesRegex(ValueError, "inference pipeline path"),
            ):
                register_models_from_report(report_path)


if __name__ == "__main__":
    unittest.main()
