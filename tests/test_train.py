from sklearn.linear_model import LogisticRegression

from heart_disease import train


def test_evaluate_predictions_returns_expected_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.1, 0.8, 0.4, 0.2, 0.9]
    metrics = train.evaluate_predictions(y_true, y_pred, y_proba)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_train_and_track_smoke(tmp_path, synthetic_clean_df, monkeypatch):
    """End-to-end smoke test with a tiny search space and a temp MLflow store."""
    tiny_search_space = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=200),
            "param_grid": {"classifier__C": [1.0]},
        }
    }
    monkeypatch.setattr(train, "MODEL_SEARCH_SPACE", tiny_search_space)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "FIGURES_DIR", tmp_path / "figures")

    result = train.train_and_track(
        synthetic_clean_df,
        experiment_name="test-experiment",
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
    )

    assert result["best_model_name"] == "logistic_regression"
    assert (tmp_path / "models" / "best_model.joblib").exists()
    assert (tmp_path / "mlruns").exists()

    metrics = result["results"]["logistic_regression"]["metrics"]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
