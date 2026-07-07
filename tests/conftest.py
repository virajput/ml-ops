import numpy as np
import pandas as pd
import pytest

from heart_disease.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


@pytest.fixture
def synthetic_clean_df() -> pd.DataFrame:
    """A small, already-clean dataframe matching the processed dataset schema."""
    rng = np.random.RandomState(0)
    n = 60
    data = {
        "age": rng.randint(29, 77, n),
        "trestbps": rng.randint(94, 200, n),
        "chol": rng.randint(126, 564, n),
        "thalach": rng.randint(71, 202, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "sex": rng.randint(0, 2, n),
        "cp": rng.randint(1, 5, n),
        "fbs": rng.randint(0, 2, n),
        "restecg": rng.randint(0, 3, n),
        "exang": rng.randint(0, 2, n),
        "slope": rng.randint(1, 4, n),
        "ca": rng.randint(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
        TARGET: rng.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) == set(df.columns)
    return df
