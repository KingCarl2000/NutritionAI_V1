import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class GeneratedModel(pa.DataFrameModel):

    class Config:
        coerce = True
        strict = True

    User_ID: Series[int] = pa.Field(ge=10001159.0, le=19999394.0)
    Gender: Series[str]
    Age: Series[int] = pa.Field(ge=20.0, le=79.0)
    Height: Series[float] = pa.Field(ge=123.0, le=218.0)
    Weight: Series[float] = pa.Field(ge=36.0, le=132.0)
    Duration: Series[float] = pa.Field(ge=1.0, le=30.0)
    Heart_Rate: Series[float] = pa.Field(ge=67.0, le=128.0)
    Body_Temp: Series[float] = pa.Field(ge=37.1, le=41.5)
    Calories: Series[float] = pa.Field(ge=1.0, le=314.0)
