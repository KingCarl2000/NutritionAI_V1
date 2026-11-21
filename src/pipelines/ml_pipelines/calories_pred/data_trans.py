"""Transform validated calories datasets and persist DVC/MLflow artifacts."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer

from src.data_io.dvc_handler import DVCHandler
from src.nutrition_core.exception.exception import DataTransformationError
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.preprocessing import (
    build_preprocessing_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "pipelines"
    / "ml_calories"
    / "data_transformation.yaml"
)
EXPECTED_SPLITS = ("train", "val", "test")


def transform_splits(
    preprocessor: ColumnTransformer,
    feature_splits: Mapping[str, pd.DataFrame],
    target_splits: Mapping[str, pd.Series],
) -> Dict[str, np.ndarray]:
    """Fit only on training features, then transform the other splits."""
    transformed_train = preprocessor.fit_transform(feature_splits["train"])
    transformed = {"train": transformed_train}
    for split in ("val", "test"):
        transformed[split] = preprocessor.transform(feature_splits[split])

    return {
        split: np.column_stack(
            (
                np.asarray(transformed[split], dtype=np.float64),
                pd.to_numeric(target_splits[split], errors="raise").to_numpy(
                    dtype=np.float64
                ),
            )
        )
        for split in EXPECTED_SPLITS
    }


def verify_gatekeeper_reports(
    validation_report_path: Path, drift_report_path: Path
) -> Dict[str, Any]:
    """Fail closed unless validation and the reported dataset drift both pass."""
    with validation_report_path.open("r", encoding="utf-8") as report_file:
        validation_report = json.load(report_file)
    with drift_report_path.open("r", encoding="utf-8") as report_file:
        drift_report = json.load(report_file)

    if not isinstance(validation_report, dict):
        raise DataTransformationError("Validation report must be a JSON object.")
    if validation_report.get("status") != "PASSED":
        raise DataTransformationError(
            "Validation gatekeeper did not report status PASSED."
        )
    for split in EXPECTED_SPLITS:
        split_report = validation_report.get(split)
        if not isinstance(split_report, dict) or split_report.get("status") != "PASSED":
            raise DataTransformationError(
                f"Validation gatekeeper did not pass the '{split}' split."
            )
        if split_report.get("error_count", 0) != 0:
            raise DataTransformationError(
                f"Validation gatekeeper reported errors for '{split}'."
            )

    if not isinstance(drift_report, dict):
        raise DataTransformationError("Drift report must be a JSON object.")
    if drift_report.get("status") not in (None, "PASSED"):
        raise DataTransformationError(
            f"Drift gatekeeper status is {drift_report.get('status')!r}."
        )

    drift_metrics = [
        metric
        for metric in drift_report.get("metrics", [])
        if isinstance(metric, dict)
        and (
            metric.get("metric") == "DriftedColumnsCount"
            or str(metric.get("metric_name", "")).startswith(
                "DriftedColumnsCount("
            )
        )
    ]
    if len(drift_metrics) != 1:
        raise DataTransformationError(
            "Drift report must contain exactly one DriftedColumnsCount metric."
        )

    drift_metric = drift_metrics[0]
    drift_value = drift_metric.get("result", drift_metric.get("value"))
    if not isinstance(drift_value, dict):
        raise DataTransformationError(
            "DriftedColumnsCount metric does not contain a result object."
        )
    if isinstance(drift_value.get("dataset_drift"), bool):
        dataset_drift = drift_value["dataset_drift"]
        drift_share = drift_value.get("share")
    else:
        drift_share = drift_value.get("share", drift_value.get("drift_share"))
        drift_threshold = drift_metric.get("config", {}).get("drift_share")
        if drift_share is None or drift_threshold is None:
            raise DataTransformationError(
                "Drift report is missing dataset drift status or share threshold."
            )
        try:
            drift_share = float(drift_share)
            drift_threshold = float(drift_threshold)
        except (TypeError, ValueError) as exc:
            raise DataTransformationError(
                "Drift report contains a non-numeric share or threshold."
            ) from exc
        if (
            not math.isfinite(drift_share)
            or not math.isfinite(drift_threshold)
            or not 0 <= drift_share <= 1
            or not 0 <= drift_threshold <= 1
        ):
            raise DataTransformationError(
                "Drift report share and threshold must be between 0 and 1."
            )
        dataset_drift = drift_share >= drift_threshold

    if dataset_drift:
        raise DataTransformationError(
            f"Drift gatekeeper detected dataset drift (share={drift_share})."
        )

    return {
        "validation_status": "PASSED",
        "drift_status": "PASSED",
        "drift_share": drift_share,
    }


class CaloriesDataTransformation:
    """Load configuration, validate gatekeepers, and transform all data splits."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        with self.config_path.open("r", encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file) or {}

        self.dvc_handler = DVCHandler(repo=str(PROJECT_ROOT))
        self.artifacts = self.config["artifacts"]
        self.column_mapping = self.config["column_mapping"]
        self.preprocessing_config = self.config["preprocessing"]

    def _resolve_path(self, configured_path: str) -> Path:
        path = Path(configured_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _load_datasets(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        datasets: Dict[str, pd.DataFrame] = {}
        for split in EXPECTED_SPLITS:
            configured_path = self.artifacts[f"valid_{split}_path"]
            local_path = self._resolve_path(configured_path).resolve()
            try:
                dvc_path = local_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                datasets[split] = pd.read_csv(local_path)
            else:
                datasets[split] = self.dvc_handler.get_dataset(
                    data_path=dvc_path,
                    format_type="csv",
                    fallback_path=str(local_path),
                )

        feature_columns = list(self.column_mapping["numerical_features"]) + list(
            self.column_mapping["categorical_features"]
        )
        target_column = self.column_mapping["target"]
        id_column = self.column_mapping.get("id_column")
        required_columns = set(feature_columns + [target_column])
        missing_values: Dict[str, int] = {}

        for split, dataset in datasets.items():
            if dataset.empty:
                raise DataTransformationError(
                    f"The validated '{split}' dataset is empty."
                )
            missing_columns = required_columns.difference(dataset.columns)
            if missing_columns:
                raise DataTransformationError(
                    f"The '{split}' dataset is missing required columns: "
                    f"{sorted(missing_columns)}"
                )
            if id_column:
                dataset.drop(columns=[id_column], errors="ignore", inplace=True)

            missing_values[split] = int(
                dataset[feature_columns + [target_column]].isna().sum().sum()
            )
            if dataset[target_column].isna().any():
                raise DataTransformationError(
                    f"The '{split}' target column contains missing values."
                )

        return datasets, missing_values

    def _split_features_and_target(
        self, datasets: Mapping[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        feature_columns = list(self.column_mapping["numerical_features"]) + list(
            self.column_mapping["categorical_features"]
        )
        target_column = self.column_mapping["target"]
        features = {
            split: dataset.loc[:, feature_columns]
            for split, dataset in datasets.items()
        }
        targets = {
            split: dataset.loc[:, target_column]
            for split, dataset in datasets.items()
        }
        return features, targets

    @staticmethod
    def _loggable_params(config: Mapping[str, Any]) -> Dict[str, str]:
        params: Dict[str, str] = {}
        for section in ("column_mapping", "preprocessing"):
            for name, value in config[section].items():
                params[f"{section}.{name}"] = json.dumps(
                    value, sort_keys=True, default=str
                )
        return params

    def run(self) -> Dict[str, Any]:
        """Run transformation and persist outputs plus MLflow lineage."""
        validation_path = self._resolve_path(self.artifacts["valid_report_path"])
        drift_path = self._resolve_path(self.artifacts["drift_report_path"])
        gatekeeper_status = verify_gatekeeper_reports(validation_path, drift_path)

        tracker = MLOpsTracker(
            experiment_name=self.config["mlflow"]["experiment_name"]
        )
        with tracker.start_run(run_name=self.config["mlflow"]["run_name"]):
            datasets, missing_values = self._load_datasets()
            feature_splits, target_splits = self._split_features_and_target(datasets)
            numerical_features: Sequence[str] = self.column_mapping[
                "numerical_features"
            ]
            categorical_features: Sequence[str] = self.column_mapping[
                "categorical_features"
            ]
            preprocessor = build_preprocessing_pipeline(
                self.preprocessing_config,
                numerical_features,
                categorical_features,
            )
            transformed = transform_splits(
                preprocessor, feature_splits, target_splits
            )

            output_paths = {
                split: self._resolve_path(
                    self.artifacts[f"transformed_{split}_path"]
                )
                for split in EXPECTED_SPLITS
            }
            preprocessor_path = self._resolve_path(
                self.artifacts["preprocessor_obj_path"]
            )
            report_path = self._resolve_path(
                self.artifacts["transformed_report_path"]
            )
            for path in [*output_paths.values(), preprocessor_path, report_path]:
                path.parent.mkdir(parents=True, exist_ok=True)

            for split, array in transformed.items():
                np.save(output_paths[split], array)
            joblib.dump(preprocessor, preprocessor_path)

            feature_names = preprocessor.get_feature_names_out().tolist()
            report = {
                "pipeline_name": "calories_data_transformation",
                "status": "SUCCESS",
                "gatekeepers": gatekeeper_status,
                "parameters": {
                    "column_mapping": self.column_mapping,
                    "preprocessing": self.preprocessing_config,
                },
                "missing_values_before_imputation": missing_values,
                "dropped_identifier": self.column_mapping.get("id_column"),
                "transformed_feature_names": feature_names,
                "output_shapes": {
                    split: list(array.shape) for split, array in transformed.items()
                },
                "artifacts": {
                    split: str(path) for split, path in output_paths.items()
                },
                "preprocessor": str(preprocessor_path),
            }
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False)

            if self.config["mlflow"].get("log_params", True):
                tracker.log_params_and_metrics(
                    params=self._loggable_params(self.config)
                )
            if self.config["mlflow"].get("log_data_shapes", True):
                metrics = {
                    f"{split}_rows": float(array.shape[0])
                    for split, array in transformed.items()
                }
                metrics.update(
                    {
                        f"{split}_columns": float(array.shape[1])
                        for split, array in transformed.items()
                    }
                )
                tracker.log_params_and_metrics(metrics=metrics)

            mlflow.log_artifact(str(report_path))
            if self.config["mlflow"].get("register_preprocessor", True):
                mlflow.log_artifact(str(preprocessor_path))

            logger.info("Calories data transformation completed successfully.")
            return report


if __name__ == "__main__":
    CaloriesDataTransformation().run()