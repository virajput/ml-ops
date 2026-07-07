import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from heart_disease.data import TARGET
from heart_disease.features import build_pipeline
from heart_disease.predict import FEATURE_ORDER, HeartDiseaseModel


@pytest.fixture
def trained_model_path(tmp_path, synthetic_clean_df):
    X = synthetic_clean_df.drop(columns=[TARGET])
    y = synthetic_clean_df[TARGET]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    return path, X


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartDiseaseModel(tmp_path / "does_not_exist.joblib")


def test_predict_one_returns_prediction_and_probability(trained_model_path):
    path, X = trained_model_path
    model = HeartDiseaseModel(path)
    sample = {col: X.iloc[0][col] for col in FEATURE_ORDER}

    result = model.predict_one(sample)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_one_missing_feature_raises(trained_model_path):
    path, X = trained_model_path
    model = HeartDiseaseModel(path)
    sample = {col: X.iloc[0][col] for col in FEATURE_ORDER}
    del sample["age"]

    with pytest.raises(ValueError, match="age"):
        model.predict_one(sample)
