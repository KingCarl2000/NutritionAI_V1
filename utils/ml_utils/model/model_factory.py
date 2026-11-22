from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
from optuna import Trial
from optuna.exceptions import TrialPruned
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


class BaseModelWrapper:
    """Provide one fit/predict/persistence API for supported estimators."""

    def __init__(
        self,
        estimator: Any,
        model_name: str,
        pruning_metric: str = "rmse",
    ) -> None:
        self.estimator = estimator
        self.model_name = model_name
        self.pruning_metric = pruning_metric

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        trial: Optional[Trial] = None,
        **fit_params: Any,
    ) -> BaseModelWrapper:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if self._is_xgboost_estimator() and trial is not None and X_val is None:
            raise ValueError(
                "X_val and y_val are required to prune an XGBoost trial."
            )

        if self._is_xgboost_estimator() and X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params.setdefault("verbose", False)
            if trial is not None:
                callback = self._create_xgboost_pruning_callback(trial)
                self.estimator.set_params(callbacks=[callback])
                try:
                    self.estimator.fit(X, y, **fit_params)
                finally:
                    self.estimator.set_params(callbacks=[])
                return self

        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        predict_proba: Optional[Callable[[Any], Any]] = getattr(
            self.estimator, "predict_proba", None
        )
        if predict_proba is None:
            raise NotImplementedError(
                f"{self.model_name} does not support probability predictions."
            )
        return predict_proba(X)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> BaseModelWrapper:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Expected a {cls.__name__} at {path}, got "
                f"{type(loaded).__name__}."
            )
        return loaded

    def _is_xgboost_estimator(self) -> bool:
        return type(self.estimator).__module__.startswith("xgboost.")

    def _create_xgboost_pruning_callback(self, trial: Trial) -> Any:
        from xgboost.callback import TrainingCallback

        metric_name = self.pruning_metric

        class OptunaPruningCallback(TrainingCallback):
            def after_iteration(
                self,
                model: Any,
                epoch: int,
                evals_log: Any,
            ) -> bool:
                del model
                validation_metrics = evals_log.get("validation_0", {})
                if metric_name not in validation_metrics:
                    available = ", ".join(validation_metrics)
                    raise ValueError(
                        f"XGBoost validation metric {metric_name!r} was not "
                        f"reported; available metrics: {available or 'none'}."
                    )

                trial.report(float(validation_metrics[metric_name][-1]), step=epoch)
                if trial.should_prune():
                    raise TrialPruned(
                        f"Trial pruned at boosting iteration {epoch}."
                    )
                return False

        return OptunaPruningCallback()


class ModelFactory:
    """Create reproducible regression estimators and suggest their parameters."""

    _MODEL_ALIASES = {
        "ridge": "ridge",
        "random_forest": "random_forest",
        "random_forest_regressor": "random_forest",
        "gradient_boosting": "gradient_boosting",
        "gradient_boosting_regressor": "gradient_boosting",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "xgb_regressor": "xgboost",
        "xgboost_regressor": "xgboost",
    }

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return ModelFactory._MODEL_ALIASES[normalized]
        except KeyError as error:
            supported = ", ".join(sorted(ModelFactory._MODEL_ALIASES))
            raise ValueError(
                f"Unsupported model {model_name!r}. Supported models: {supported}."
            ) from error

    @staticmethod
    def suggest_hyperparameters(
        trial: Trial,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if model_name is None:
            model_name = str(
                trial.suggest_categorical(
                    "model_name",
                    ["ridge", "random_forest", "gradient_boosting", "xgboost"],
                )
            )
        name = ModelFactory._normalize_model_name(model_name)

        if name == "ridge":
            return {
                "alpha": trial.suggest_float("ridge.alpha", 1e-4, 100.0, log=True),
            }
        if name == "random_forest":
            return {
                "n_estimators": trial.suggest_int(
                    "random_forest.n_estimators", 100, 500, step=50
                ),
                "max_depth": trial.suggest_int(
                    "random_forest.max_depth", 3, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "random_forest.min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_float(
                    "random_forest.max_features", 0.5, 1.0
                ),
            }
        if name == "gradient_boosting":
            return {
                "n_estimators": trial.suggest_int(
                    "gradient_boosting.n_estimators", 50, 300, step=25
                ),
                "learning_rate": trial.suggest_float(
                    "gradient_boosting.learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int(
                    "gradient_boosting.max_depth", 2, 8
                ),
                "subsample": trial.suggest_float(
                    "gradient_boosting.subsample", 0.6, 1.0
                ),
            }
        return {
            "n_estimators": trial.suggest_int(
                "xgboost.n_estimators", 100, 600, step=50
            ),
            "max_depth": trial.suggest_int("xgboost.max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "xgboost.learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("xgboost.subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "xgboost.colsample_bytree", 0.6, 1.0
            ),
        }

    @staticmethod
    def create_model(
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseModelWrapper:
        name = ModelFactory._normalize_model_name(model_name)
        model_params = dict(params or {})

        if name == "ridge":
            estimator: BaseEstimator = Ridge(**model_params)
        elif name == "random_forest":
            model_params.setdefault("random_state", 42)
            model_params.setdefault("n_jobs", -1)
            estimator = RandomForestRegressor(**model_params)
        elif name == "gradient_boosting":
            model_params.setdefault("random_state", 42)
            estimator = GradientBoostingRegressor(**model_params)
        else:
            from xgboost import XGBRegressor

            model_params.setdefault("random_state", 42)
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("objective", "reg:squarederror")
            model_params.setdefault("eval_metric", "rmse")
            estimator = XGBRegressor(**model_params)

        return BaseModelWrapper(estimator=estimator, model_name=name)

    @staticmethod
    def create_model_from_trial(
        trial: Trial,
        model_name: Optional[str] = None,
    ) -> Tuple[BaseModelWrapper, str]:
        if model_name is None:
            model_name = str(
                trial.suggest_categorical(
                    "model_name",
                    ["ridge", "random_forest", "gradient_boosting", "xgboost"],
                )
            )
        normalized_name = ModelFactory._normalize_model_name(model_name)
        params = ModelFactory.suggest_hyperparameters(trial, normalized_name)
        return ModelFactory.create_model(normalized_name, params), normalized_name