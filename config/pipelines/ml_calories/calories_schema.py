import pandera.pandas as pa
from pandera.typing import Series

from src.nutrition_core.pandera.api.base import BaseSchema


class MlCaloriesInputSchema(BaseSchema):
    """Calories input schema with strict, pipeline-specific business constraints."""

    class Config(BaseSchema.Config):
        coerce = True
        strict = True

    User_ID: Series[int] = pa.Field(ge=1)
    Gender: Series[str] = pa.Field(isin=["male", "female", "other"])
    Age: Series[int] = pa.Field(ge=0, le=200)
    Height: Series[float] = pa.Field(ge=40, le=255.0)
    Weight: Series[float] = pa.Field(ge=2, le=300.0)
    Duration: Series[float] = pa.Field(ge=0, le=300.0)
    Heart_Rate: Series[float] = pa.Field(ge=40.0, le=220.0)
    Body_Temp: Series[float] = pa.Field(ge=35.0, le=42.5)


class MlCaloriesTrainingSchema(MlCaloriesInputSchema):
    """Training schema: calorie input schema plus target-column constraints."""

    Calories: Series[float] = pa.Field(ge=1.0, le=5000.0)
