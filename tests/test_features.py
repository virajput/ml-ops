from sklearn.linear_model import LogisticRegression

from heart_disease.data import TARGET
from heart_disease.features import build_pipeline, build_preprocessor


def test_preprocessor_transforms_without_error(synthetic_clean_df):
    X = synthetic_clean_df.drop(columns=[TARGET])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_pipeline_fits_and_predicts(synthetic_clean_df):
    X = synthetic_clean_df.drop(columns=[TARGET])
    y = synthetic_clean_df[TARGET]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    assert set(preds).issubset({0, 1})
    assert proba.shape == (len(X), 2)
