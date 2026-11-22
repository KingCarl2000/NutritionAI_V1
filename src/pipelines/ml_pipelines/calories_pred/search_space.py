from typing import Dict


def get_search_space(trial) -> Dict[str, object]:
    """Define conditional Optuna parameters for Calories model training."""
    model_type = trial.suggest_categorical(
        "train_params.model_type",
        ["ridge", "random_forest", "xgboost"],
    )

    search_space: Dict[str, object] = {
        "train_params.model_type": model_type,
        "train_params.batch_size": trial.suggest_categorical(
            "train_params.batch_size",
            [16, 32, 64],
        ),
    }

    if model_type == "ridge":
        search_space["train_params.learning_rate"] = trial.suggest_float(
            "train_params.learning_rate",
            0.0001,
            0.1,
            log=True,
        )
    elif model_type == "random_forest":
        search_space["train_params.n_estimators"] = trial.suggest_int(
            "train_params.n_estimators",
            50,
            500,
            step=50,
        )
        search_space["train_params.max_depth"] = trial.suggest_int(
            "train_params.max_depth",
            4,
            10,
        )
    else:
        search_space["train_params.n_estimators"] = trial.suggest_int(
            "train_params.n_estimators",
            50,
            500,
            step=50,
        )
        search_space["train_params.learning_rate"] = trial.suggest_float(
            "train_params.learning_rate",
            0.0001,
            0.1,
            log=True,
        )

    return search_space
