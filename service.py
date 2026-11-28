"""BentoML service for raw-input calories prediction."""

from typing import Any, Literal

import bentoml
from bentoml.exceptions import BentoMLException
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Target model tag in BentoML Model Store
MODEL_TAG = "calories_predictor_r2_score:latest"

# Minimal Debian-based runtime image (BentoML automatically injects its matching version)
runtime_image = bentoml.images.Image(
    python_version="3.12",
    distro="debian",
).python_packages(
    "numpy==1.26.4",
    "pandas==2.3.3",
    "pydantic==2.13.4",
    "scikit-learn==1.9.0",
)


class CaloriesPredictionRequest(BaseModel):
    """Validated raw feature values expected by the calories model."""

    model_config = ConfigDict(extra="forbid")

    Gender: Literal["male", "female", "other"]
    Age: int = Field(ge=0, le=200)
    Weight: float = Field(ge=2, le=300)
    Height: float = Field(ge=40, le=255)
    Duration: float = Field(ge=0, le=300)
    Heart_Rate: float = Field(ge=40, le=220)
    Body_Temp: float = Field(ge=35, le=42.5)


class CaloriesPredictionResponse(BaseModel):
    """Prediction output schema."""

    Calories: float


NUMERICAL_FEATURES = (
    "Age",
    "Weight",
    "Height",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
)
FEATURES = (*NUMERICAL_FEATURES, "Gender")


def _raw_features(request: CaloriesPredictionRequest) -> pd.DataFrame:
    """Create the typed raw DataFrame expected by the trained inference pipeline."""
    data = request.model_dump()
    frame = pd.DataFrame([data], columns=FEATURES)
    for column in NUMERICAL_FEATURES:
        frame[column] = frame[column].astype("float64")
    return frame


@bentoml.service(
    name="calories-prediction",
    image=runtime_image,
    resources={"cpu": "1", "memory": "1Gi"},
    traffic={"timeout": 30},
)
class CaloriesPredictionService:
    """HTTP inference service wrapping the R2-selected calories prediction model."""

    model_ref = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        self.model: Any = bentoml.sklearn.load_model(self.model_ref)

    @bentoml.api
    def predict(self, request: CaloriesPredictionRequest) -> CaloriesPredictionResponse:
        """Predict burned calories from validated user and workout metrics."""
        try:
            features = _raw_features(request)
            predictions = np.asarray(self.model.predict(features)).reshape(-1)
        except Exception as exc:
            raise BentoMLException(
                f"Failed to execute calories prediction inference: {exc}"
            ) from exc

        if predictions.size != 1 or not np.isfinite(predictions[0]):
            raise BentoMLException("Calories model returned an invalid prediction.")

        return CaloriesPredictionResponse(Calories=float(predictions[0]))
