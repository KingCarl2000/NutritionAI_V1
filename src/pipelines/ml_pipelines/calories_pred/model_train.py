"""Hydra entrypoint for Optuna-based Calories model training."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig
from sqlalchemy.engine import make_url
from sklearn.metrics import mean_squared_error, r2_score

from src.nutrition_core.logging.logger import logger
from utils.ml_utils.model.model_trainer import ModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = "../../../../config/pipelines/ml_calories"

METRICS = {
    "mean_squared_error": (
        lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred)),
        "minimize",
    ),
    "root_mean_squared_error": (
        lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred) ** 0.5),
        "minimize",
    ),
    "root_mean_squared_log_error": (
        lambda y_true, y_pred: float(
            mean_squared_error(
                np.log1p(np.maximum(y_true, 0)),
                np.log1p(np.maximum(y_pred, 0)),
            )
            ** 0.5
        ),
        "minimize",
    ),
    "r2_score": (
        lambda y_true, y_pred: float(r2_score(y_true, y_pred)),
        "maximize",
    ),
}


def _resolve_project_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optuna_storage(storage_url: str | None) -> str | None:
    if not storage_url:
        return storage_url

    url = make_url(storage_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return storage_url
    if url.database == ":memory:" or url.database.startswith("file:"):
        return storage_url

    database_path = _resolve_project_path(url.database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return url.set(database=database_path.as_posix()).render_as_string(
        hide_password=False
    )


def _load_transformed_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Transformed dataset was not found: {path}")

    data = np.load(path, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected a 2D transformed dataset with features and target: {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Transformed dataset contains non-finite values: {path}")

    return data[:, :-1], data[:, -1]


def run_training(cfg: DictConfig) -> Dict[str, Any]:
    """Load transformed train/validation data, run HPO, and write DVC metrics."""
    artifacts = cfg.train_params.artifacts
    train_path = _resolve_project_path(str(artifacts.transformed_train_path))
    val_path = _resolve_project_path(str(artifacts.transformed_val_path))
    preprocessor_path = _resolve_project_path(str(artifacts.preprocessor_path))
    model_path = _resolve_project_path(str(artifacts.best_model_path))
    report_path = _resolve_project_path(str(artifacts.training_report_path))

    X_train, y_train = _load_transformed_split(train_path)
    X_val, y_val = _load_transformed_split(val_path)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Training and validation datasets have different feature counts."
        )
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Preprocessor was not found: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    expected_feature_count = len(preprocessor.get_feature_names_out())
    if X_train.shape[1] != expected_feature_count:
        raise ValueError(
            "Transformed feature count does not match the saved preprocessor: "
            f"{X_train.shape[1]} != {expected_feature_count}."
        )

    configured_model_name = str(cfg.training.model_name).strip().lower()
    model_name = (
        None if configured_model_name in {"", "auto", "none"} else configured_model_name
    )

    configured_metric = str(cfg.training.metric_name).strip().lower()
    if configured_metric in {"", "auto", "none"}:
        metric_names = [str(name).strip().lower() for name in cfg.training.metric_names]
    else:
        metric_names = [configured_metric]
    if not metric_names:
        raise ValueError("At least one metric_name must be configured.")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("Configured metric_names must be unique.")

    configured_direction = str(cfg.training.direction).strip().lower()
    optuna_config = cfg.get("optuna", {})
    storage_url = _resolve_optuna_storage(optuna_config.get("storage"))
    study_name_prefix = optuna_config.get("study_name")
    metric_results: Dict[str, Any] = {}
    for metric_name in metric_names:
        if metric_name not in METRICS:
            raise ValueError(
                f"Unsupported metric_name {metric_name!r}. "
                f"Choose from: {', '.join(METRICS)}."
            )
        eval_metric_fn, metric_direction = METRICS[metric_name]
        if configured_direction not in {"auto", metric_direction}:
            raise ValueError(
                f"{metric_name} requires direction={metric_direction!r}, "
                f"not {configured_direction!r}."
            )

        metric_model_path = model_path
        if len(metric_names) > 1:
            metric_model_path = model_path.with_name(
                f"{model_path.stem}_{metric_name}{model_path.suffix}"
            )
        trainer = ModelTrainer(
            experiment_name=str(cfg.training.experiment_name),
            direction=metric_direction,
            eval_metric_fn=eval_metric_fn,
            tracking_uri=str(cfg.training.tracking_uri),
            seed=int(cfg.training.seed),
            metric_name=metric_name,
            storage=storage_url,
        )
        best_wrapper, best_params, best_score = trainer.optimize_and_train(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=int(cfg.training.n_trials),
            study_name=(
                f"{study_name_prefix}_{metric_name}" if study_name_prefix else None
            ),
            n_jobs=int(cfg.training.n_jobs),
            model_name=model_name,
            model_output_path=metric_model_path,
        )

        study = trainer.study
        if study is None:
            raise RuntimeError(
                f"ModelTrainer finished without an Optuna study for {metric_name}."
            )
        metric_results[metric_name] = {
            "direction": metric_direction,
            "best_score": best_score,
            "best_params": best_params,
            "model_type": best_wrapper.model_name,
            "model_path": str(metric_model_path),
            "trials": {
                "total": len(study.trials),
                "completed": sum(
                    trial.state.name == "COMPLETE" for trial in study.trials
                ),
                "pruned": sum(trial.state.name == "PRUNED" for trial in study.trials),
                "failed": sum(trial.state.name == "FAIL" for trial in study.trials),
            },
        }
        logger.info(
            "Calories training completed: metric=%s direction=%s score=%.6f",
            metric_name,
            metric_direction,
            best_score,
        )

    report: Dict[str, Any] = {
        "pipeline_name": "calories_model_training",
        "status": "SUCCESS",
        "metric_results": metric_results,
        "data": {
            "train_rows": int(X_train.shape[0]),
            "validation_rows": int(X_val.shape[0]),
            "feature_count": int(X_train.shape[1]),
        },
        "artifacts": {
            "models": {
                metric_name: result["model_path"]
                for metric_name, result in metric_results.items()
            },
            "preprocessor": str(preprocessor_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="hydra_config",
)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
