import pandera.pandas as pa
from pandera.typing import Series

from src.nutrition_core.pandera.api.base import BaseSchema
# Importing this module registers the custom checks as Pandera Field checks.
from src.nutrition_core.pandera.api import check as _custom_checks  # noqa: F401


class MlCaloriesInputSchema(BaseSchema):
    """Calories input schema with strict, pipeline-specific business constraints."""

    class Config(BaseSchema.Config):
        coerce = True
        strict = True

    User_ID: Series[int] = pa.Field(
        is_in_numeric_range={"min_val": 1, "max_val": 2**63 - 1}
    )
    Gender: Series[str] = pa.Field(
        isin=["male", "female", "other"],
        is_valid_string_length={"min_len": 1, "max_len": 10},
    )
    Age: Series[int] = pa.Field(
        is_in_numeric_range={"min_val": 0, "max_val": 200}
    )
    Height: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 40, "max_val": 255.0}
    )
    Weight: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 2, "max_val": 300.0}
    )
    Duration: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 0, "max_val": 300.0}
    )
    Heart_Rate: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 40.0, "max_val": 220.0}
    )
    Body_Temp: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 35.0, "max_val": 42.5}
    )


class MlCaloriesTrainingSchema(MlCaloriesInputSchema):
    """Training schema: calorie input schema plus target-column constraints."""

    Calories: Series[float] = pa.Field(
        is_in_numeric_range={"min_val": 1.0, "max_val": 5000.0}
    )
