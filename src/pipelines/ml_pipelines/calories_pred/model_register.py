"""Import MLflow-logged calories models into the local BentoML model store."""

import argparse
import json
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
    """Import each logged metric model using its MLflow model URI."""
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

    imported_models: Dict[str, Any] = {}
    for metric_name, result in metric_results.items():
        model_uri = result.get("mlflow_model_uri") if isinstance(result, dict) else None
        if not isinstance(model_uri, str) or not model_uri:
            raise ValueError(
                f"Training report is missing the MLflow model URI "
                f"for metric {metric_name!r}."
            )

        model_name = f"{model_name_prefix}_{metric_name}"
        if not re.fullmatch(r"[a-z0-9_]+", model_name):
            raise ValueError(f"Invalid BentoML model name: {model_name!r}")

        imported_models[metric_name] = bentoml.mlflow.import_model(
            name=model_name,
            model_uri=model_uri,
            metadata={"mlflow_run_id": run_id, "metric": metric_name},
        )

    return imported_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model-name-prefix", default="calories_predictor")
    args = parser.parse_args()

    imported_models = register_models_from_report(
        report_path=args.report,
        model_name_prefix=args.model_name_prefix,
    )
    for metric_name, model in imported_models.items():
        print(f"Imported {metric_name}: {model}")


if __name__ == "__main__":
    main()
