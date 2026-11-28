"""Register DVC-tracked inference pipelines in the BentoML model store."""

import argparse
import json
import joblib
import re
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "Artifacts"
    / "reports"
    / "ml_calories"
    / "training"
    / "training_report.json"
)


def register_models_from_report(
    report_path: str | Path = DEFAULT_REPORT_PATH,
    model_name_prefix: str = "calories_predictor",
) -> Dict[str, Any]:
    """Register each saved preprocessing-and-model pipeline with BentoML."""
    report_file = Path(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(f"Training report was not found: {report_file}")

    report = json.loads(report_file.read_text(encoding="utf-8"))
    run_id = report.get("run_id")
    metric_results = report.get("metric_results")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Training report is missing a valid MLflow run_id.")
    if not isinstance(metric_results, dict) or not metric_results:
        raise ValueError("Training report does not contain any metric models.")
    if not re.fullmatch(r"[a-z0-9_]+", model_name_prefix):
        raise ValueError(
            "model_name_prefix must contain lowercase letters, digits, or underscores."
        )

    import bentoml

    registered_models: Dict[str, Any] = {}
    for metric_name, result in metric_results.items():
        pipeline_path = (
            result.get("inference_pipeline_path") if isinstance(result, dict) else None
        )
        if not isinstance(pipeline_path, str) or not pipeline_path:
            raise ValueError(
                f"Training report is missing the inference pipeline path "
                f"for metric {metric_name!r}."
            )

        pipeline_file = Path(pipeline_path).expanduser()
        if not pipeline_file.is_absolute():
            pipeline_file = PROJECT_ROOT / pipeline_file
        if not pipeline_file.is_file():
            raise FileNotFoundError(
                f"Inference pipeline for metric {metric_name!r} was not found: "
                f"{pipeline_file}"
            )

        model_name = f"{model_name_prefix}_{metric_name}"
        if not re.fullmatch(r"[a-z0-9_]+", model_name):
            raise ValueError(f"Invalid BentoML model name: {model_name!r}")

        registered_models[metric_name] = bentoml.sklearn.save_model(
            name=model_name,
            model=joblib.load(pipeline_file),
            metadata={"training_run_id": run_id, "metric": metric_name},
        )

    return registered_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model-name-prefix", default="calories_predictor")
    args = parser.parse_args()

    registered_models = register_models_from_report(
        report_path=args.report,
        model_name_prefix=args.model_name_prefix,
    )
    for metric_name, model in registered_models.items():
        print(f"Registered {metric_name}: {model}")


if __name__ == "__main__":
    main()
