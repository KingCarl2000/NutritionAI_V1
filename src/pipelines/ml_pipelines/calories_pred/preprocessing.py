"""Preprocessing pipeline construction for the calories prediction dataset."""

from typing import Any, Mapping, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def build_preprocessing_pipeline(
    config: Mapping[str, Any],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Build the configured numeric and categorical preprocessing branches."""
    numerical_config = config.get("numerical", {})
    imputer_config = numerical_config.get("imputer", {})
    imputer_type = imputer_config.get("type", "simple").lower()

    if imputer_type == "knn":
        numerical_imputer = KNNImputer(
            n_neighbors=imputer_config.get("n_neighbors", 5),
            weights=imputer_config.get("weights", "uniform"),
        )
    elif imputer_type == "simple":
        numerical_imputer = SimpleImputer(
            strategy=imputer_config.get("strategy", "median")
        )
    else:
        raise ValueError(f"Unsupported numerical imputer type: {imputer_type}")

    scaler_config = numerical_config.get("scaler", {})
    scaler_type = scaler_config.get("type", "robust").lower()
    scaler_options = {
        key: value for key, value in scaler_config.items() if key != "type"
    }
    for tuple_param in ("quantile_range", "feature_range"):
        if isinstance(scaler_options.get(tuple_param), list):
            scaler_options[tuple_param] = tuple(scaler_options[tuple_param])

    scaler_factories = {
        "robust": RobustScaler,
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
    }
    if scaler_type == "none":
        scaler = "passthrough"
    elif scaler_type in scaler_factories:
        scaler = scaler_factories[scaler_type](**scaler_options)
    else:
        raise ValueError(f"Unsupported numerical scaler type: {scaler_type}")

    categorical_config = config.get("categorical", {})
    categorical_imputer_config = categorical_config.get("imputer", {})
    if categorical_imputer_config.get("type", "simple").lower() != "simple":
        raise ValueError(
            "Only the simple categorical imputer is supported."
        )
    categorical_imputer = SimpleImputer(
        strategy=categorical_imputer_config.get("strategy", "most_frequent"),
        **(
            {"fill_value": categorical_imputer_config["fill_value"]}
            if "fill_value" in categorical_imputer_config
            else {}
        ),
    )

    encoder_config = categorical_config.get("encoder", {})
    encoder_type = encoder_config.get("type", "onehot").lower()
    if encoder_type == "onehot":
        encoder = OneHotEncoder(
            handle_unknown=encoder_config.get("handle_unknown", "error"),
            sparse_output=encoder_config.get("sparse_output", True),
            drop=encoder_config.get("drop"),
        )
    elif encoder_type == "ordinal":
        ordinal_options = {}
        handle_unknown = encoder_config.get("handle_unknown", "error")
        if handle_unknown == "use_encoded_value":
            ordinal_options["handle_unknown"] = handle_unknown
            ordinal_options["unknown_value"] = encoder_config.get(
                "unknown_value", -1
            )
        elif handle_unknown != "error":
            raise ValueError(
                "OrdinalEncoder handle_unknown must be 'error' or "
                "'use_encoded_value'."
            )
        encoder = OrdinalEncoder(**ordinal_options)
    else:
        raise ValueError(f"Unsupported categorical encoder type: {encoder_type}")

    numerical_pipeline = Pipeline(
        [
            ("imputer", numerical_imputer),
            ("scaler", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", categorical_imputer),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        [
            ("numerical", numerical_pipeline, list(numerical_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )