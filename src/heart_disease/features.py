"""Feature engineering pipeline: scaling for numeric, one-hot encoding for categorical."""
from __future__ import annotations

import warnings

from scipy.optimize import OptimizeWarning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# With only ~300 rows, a rare category (e.g. a `restecg` value) can be absent from a given
# CV fold's training split; OneHotEncoder(handle_unknown="ignore") already handles this
# correctly by encoding it as all-zeros, so the warning is expected noise, not a bug.
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns .* during transform",
    category=UserWarning,
)

# sklearn's LogisticRegression(solver="lbfgs") passes a scipy-specific `iprint` option that
# newer scipy releases no longer recognize; scipy warns but still runs lbfgs correctly.
warnings.filterwarnings(
    "ignore",
    message="Unknown solver options: iprint",
    category=OptimizeWarning,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable ColumnTransformer: scale numeric, one-hot encode categorical."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the shared preprocessing pipeline for reproducibility."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)])
