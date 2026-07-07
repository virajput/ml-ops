"""Inference wrapper that loads the packaged pipeline (preprocessing + model)."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from heart_disease.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_model.joblib"

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class HeartDiseaseModel:
    """Loads a persisted sklearn Pipeline (preprocessing + classifier) for inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run `python -m "
                "heart_disease.train` first to produce models/best_model.joblib."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict_one(self, features: dict) -> dict:
        """Predict disease risk for a single patient record.

        `features` must contain all keys in FEATURE_ORDER.
        Returns {"prediction": 0|1, "probability": float}.
        """
        missing = set(FEATURE_ORDER) - set(features)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        row = pd.DataFrame([{col: features[col] for col in FEATURE_ORDER}])
        prediction = int(self.pipeline.predict(row)[0])
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return {"prediction": prediction, "probability": probability}
