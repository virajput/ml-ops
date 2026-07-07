import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api.main import app, model_holder
from heart_disease.data import TARGET
from heart_disease.features import build_pipeline
from heart_disease.predict import HeartDiseaseModel

VALID_PAYLOAD = {
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalach": 150,
    "oldpeak": 2.3,
    "sex": 1,
    "cp": 1,
    "fbs": 1,
    "restecg": 2,
    "exang": 0,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


@pytest.fixture
def client_with_model(tmp_path, synthetic_clean_df):
    X = synthetic_clean_df.drop(columns=[TARGET])
    y = synthetic_clean_df[TARGET]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    with TestClient(app) as client:
        model_holder["model"] = HeartDiseaseModel(model_path)
        yield client
    model_holder["model"] = None


def test_health_reports_model_loaded(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_probability(client_with_model):
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("low_risk", "high_risk")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_payload(client_with_model):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["age"]
    response = client_with_model.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client_with_model):
    model_holder["model"] = None
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_format(client_with_model):
    response = client_with_model.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text
