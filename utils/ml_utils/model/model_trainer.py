from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import mlflow
import optuna
from mlflow.tracking import MlflowClient
from optuna.trial import FixedTrial, Trial
from sklearn.metrics import mean_squared_error, r2_score

from utils.ml_utils.model.model_factory import BaseModelWrapper, ModelFactory


class ModelTrainer:
    """Orchestrate Optuna trials, nested MLflow runs, and best-model export."""

    def __init__(
        self,
        experiment_name: str,
        direction: str = "maximize",
        eval_metric_fn: Optional[Callable[[Any, Any], float]] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        tracking_uri: Optional[str] = "sqlite:///mlflow.db",
        seed: int = 42,
        metric_name: Optional[str] = None,
    ) -> None:
        normalized_direction = direction.strip().lower()
        if normalized_direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be either 'maximize' or 'minimize'.")

        self.experiment_name = experiment_name
        self.direction = normalized_direction
        self.eval_metric_fn = eval_metric_fn or (
            r2_score if normalized_direction == "maximize" else mean_squared_error
        )
        self.metric_name = metric_name or getattr(
            self.eval_metric_fn, "__name__", "custom_metric"
        )
        self.pruner = pruner or optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        )
        self.tracking_uri = tracking_uri
        self.seed = seed
        self.study: Optional[optuna.study.Study] = None
        self.best_wrapper: Optional[BaseModelWrapper] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self._parent_run_id: Optional[str] = None
        self._model_name: Optional[str] = None

    def _objective(
        self,
        trial: Trial,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        model_name: Optional[str] = None,
    ) -> float:
        if self._parent_run_id is None:
            raise RuntimeError("The objective must run inside a parent MLflow run.")

        selected_model_name = model_name or self._model_name
        child_run = mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
            parent_run_id=self._parent_run_id,
            tags={
                "trial_number": str(trial.number),
                "is_best_run": "false",
                "pruned": "false",
            },
        )
        run_status = "FINISHED"

        try:
            wrapper, selected_model_name = ModelFactory.create_model_from_trial(
                trial,
                model_name=selected_model_name,
            )
            mlflow.set_tag("model_type", selected_model_name)
            mlflow.log_params(
                {
                    **trial.params,
                    "model_type": selected_model_name,
                }
            )

            wrapper.fit(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                trial=trial,
            )
            predictions = wrapper.predict(X_val)
            score = float(self.eval_metric_fn(y_val, predictions))
            if not math.isfinite(score):
                raise ValueError(
                    f"Evaluation metric {self.metric_name!r} returned "
                    f"a non-finite value: {score}."
                )

            if not wrapper.supports_iterative_pruning:
                trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned after validation {self.metric_name}={score}."
                )

            metrics = {"val_score": score}
            if self.metric_name != "val_score":
                metrics[self.metric_name] = score
            mlflow.log_metrics(metrics)
            trial.set_user_attr("mlflow_run_id", child_run.info.run_id)
            return score
        except optuna.TrialPruned:
            run_status = "KILLED"
            mlflow.set_tags({"pruned": "true", "status": "PRUNED"})
            raise
        except Exception as error:
            run_status = "FAILED"
            mlflow.set_tags(
                {
                    "status": "FAILED",
                    "error_message": str(error)[:250],
                }
            )
            raise
        finally:
            active_run = mlflow.active_run()
            if (
                active_run is not None
                and active_run.info.run_id == child_run.info.run_id
            ):
                mlflow.end_run(status=run_status)

    def optimize_and_train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        model_name: Optional[str] = None,
        model_output_path: Optional[str | Path] = None,
        show_progress_bar: bool = False,
    ) -> Tuple[BaseModelWrapper, Dict[str, Any], float]:
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")

        self.study = None
        self.best_wrapper = None
        self.best_params = None
        self.best_score = None
        self._parent_run_id = None

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        parent_run_name = (
            f"HPO_Search_{model_name}" if model_name else "HPO_Search_AutoSelect"
        )
        nested_parent = mlflow.active_run() is not None
        with mlflow.start_run(
            run_name=parent_run_name,
            nested=nested_parent,
            tags={
                "run_type": "hpo_parent",
                "direction": self.direction,
            },
        ) as parent_run:
            self._parent_run_id = parent_run.info.run_id
            self._model_name = model_name
            mlflow.log_params(
                {
                    "direction": self.direction,
                    "n_trials": n_trials,
                    "metric_name": self.metric_name,
                    "model_selection": model_name or "automatic",
                }
            )

            self.study = optuna.create_study(
                study_name=study_name,
                direction=self.direction,
                pruner=self.pruner,
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                load_if_exists=study_name is not None,
            )
            self.study.optimize(
                lambda trial: self._objective(
                    trial,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    model_name,
                ),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress_bar,
            )

            completed_trials = [
                trial
                for trial in self.study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                raise RuntimeError(
                    "Optuna completed no trials; no best model is available."
                )

            best_trial = self.study.best_trial
            best_model_name = str(best_trial.params.get("model_name", model_name or ""))
            if not best_model_name:
                raise RuntimeError("The best Optuna trial did not record a model name.")

            self.best_wrapper, _ = ModelFactory.create_model_from_trial(
                FixedTrial(best_trial.params),
                model_name=best_model_name,
            )
            self.best_wrapper.fit(X_train, y_train)
            self.best_params = {
                **best_trial.params,
                "model_type": best_model_name,
            }
            if best_trial.value is None:
                raise RuntimeError("The best Optuna trial has no objective score.")
            self.best_score = float(best_trial.value)

            mlflow.log_metrics(
                {
                    "best_val_score": self.best_score,
                    "best_trial_number": float(best_trial.number),
                }
            )
            mlflow.log_params(
                {f"best_{key}": value for key, value in best_trial.params.items()}
            )
            mlflow.set_tags(
                {
                    "is_best_run": "true",
                    "best_trial_number": str(best_trial.number),
                    "best_model_type": best_model_name,
                }
            )

            best_child_run_id = best_trial.user_attrs.get("mlflow_run_id")
            if best_child_run_id is not None:
                MlflowClient().set_tag(
                    best_child_run_id,
                    "is_best_run",
                    "true",
                )

            with tempfile.TemporaryDirectory() as artifact_dir:
                model_artifact = Path(artifact_dir) / "best_model.pkl"
                params_artifact = Path(artifact_dir) / "best_params.json"
                self.best_wrapper.save(model_artifact)
                params_artifact.write_text(
                    json.dumps(self.best_params, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(model_artifact), artifact_path="best_model")
                mlflow.log_artifact(str(params_artifact), artifact_path="best_model")

            if model_output_path is not None:
                self.save_best_model(model_output_path)

        self._parent_run_id = None
        return self.best_wrapper, self.best_params, self.best_score

    def save_best_model(self, file_path: str | Path) -> None:
        if self.best_wrapper is None:
            raise RuntimeError("No best model is available; run optimization first.")
        self.best_wrapper.save(file_path)
