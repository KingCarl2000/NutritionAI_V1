"""
Shared Pandera base schemas and validation helpers for NutritionAI_V1.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple

import pandas as pd
import pandera.pandas as pa

from src.nutrition_core.logging.logger import logger, validation_error_handler
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.pandera.backends.pandas_executor import NutritionPandasBackend
from src.nutrition_core.exception.val_exception import DataValidationError

SchemaModelT = TypeVar("SchemaModelT", bound=pa.DataFrameModel)


class BaseSchema(pa.DataFrameModel):
    """Base model with common Pandera defaults and backend-based validation."""

    class Config:
        coerce = True
        strict = False
        ordered = False
        unique_column_names = True
        add_missing_columns = True

    @classmethod
    @validation_error_handler()
    def validate_df(
        cls: Type[SchemaModelT],
        df: pd.DataFrame,
        lazy: bool = True,
        drop_invalid_rows: bool = False,
        mlflow_tracker: Optional[MLOpsTracker] = None,
        return_errors: bool = False,
    ) -> Any:
        """Validate through the shared Pandas backend.

        Pipeline-specific callers can pass their active tracker and set
        ``return_errors=True`` to apply pipeline-specific routing/thresholds
        without bypassing the common validation implementation.
        """
        schema_name = cls.__name__
        total_rows = len(df)
        logger.info(
            f"[Pandera Base] Validating {total_rows} rows with schema '{schema_name}'."
        )

        backend = NutritionPandasBackend(
            schema=cls.to_schema(),
            lazy=lazy,
            drop_invalid_rows=drop_invalid_rows,
        )
        validated_df, errors = backend.run_pipeline(df)

        if errors:
            if mlflow_tracker:
                mlflow_tracker.log_validation_errors(schema_name, errors)

            status = "PARTIAL_SUCCESS" if drop_invalid_rows else "FAILED"
            metrics = {f"{schema_name}_error_count": float(len(errors))}
            if drop_invalid_rows:
                metrics[f"{schema_name}_dropped_rows"] = float(total_rows - len(validated_df))

            if mlflow_tracker:
                mlflow_tracker.log_params_and_metrics(
                    params={
                        f"validation_{schema_name}_status": status,
                        f"validation_{schema_name}_schema": schema_name,
                    },
                    metrics=metrics,
                )

            if not drop_invalid_rows and not return_errors:
                raise DataValidationError(
                    f"Validation failed for schema '{schema_name}'.",
                    errors=errors,
                )
        else:
            if mlflow_tracker:
                mlflow_tracker.log_params_and_metrics(
                    params={
                        f"validation_{schema_name}_status": "SUCCESS",
                        f"validation_{schema_name}_schema": schema_name,
                    },
                    metrics={f"{schema_name}_error_count": 0.0},
                )

        return (validated_df, errors) if return_errors else validated_df
