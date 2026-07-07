"""Train, tune, and compare classification models with MLflow experiment tracking."""
from __future__ import annotations

import contextlib
import logging
import warnings
from pathlib import Path

import joblib
import joblib.parallel
import matplotlib

matplotlib.use("Agg")

# Classifiers always predict integer class labels, so MLflow's output-schema inference
# reliably hints that "integer columns can't represent missing values" -- irrelevant here
# since the API validates all required fields before a request ever reaches the model.
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column",
    category=UserWarning,
)

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from tqdm.auto import tqdm  # noqa: E402

from heart_disease.data import PROCESSED_PATH, TARGET, get_clean_dataset  # noqa: E402
from heart_disease.features import build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
RANDOM_STATE = 42

MODEL_SEARCH_SPACE = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
    },
    "random_forest": {
        # n_jobs=1 pinned explicitly so it doesn't inherit the ambient threading backend's
        # n_jobs used for GridSearchCV/cross_val_score (see tqdm_joblib) -- otherwise its
        # internal per-tree parallelism also fires the progress callback and the "tuning"
        # bar overshoots its expected total.
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    },
}


@contextlib.contextmanager
def tqdm_joblib(progress_bar: tqdm):
    """Redirect joblib's (used internally by GridSearchCV/cross_val_score) batch-completion
    callbacks into a tqdm progress bar, so long-running fits report real progress."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        progress_bar.close()


def load_dataset(processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not processed_path.exists():
        return get_clean_dataset()
    return pd.read_csv(processed_path)


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_track(
    df: pd.DataFrame,
    experiment_name: str = "heart-disease-risk",
    tracking_uri: str | None = None,
) -> dict:
    """Train + tune every model in MODEL_SEARCH_SPACE, log each to MLflow, return best info."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    model_progress = tqdm(MODEL_SEARCH_SPACE.items(), desc="Training models", unit="model")
    for model_name, spec in model_progress:
        model_progress.set_description(f"Training {model_name}")
        with mlflow.start_run(run_name=model_name):
            # The "threading" backend gives real per-fit progress callbacks (unlike the
            # default sequential backend at n_jobs=1) without spawning OS subprocesses, so
            # it avoids the multiprocessing resource-tracker noise seen with a process pool.
            pipeline = build_pipeline(spec["estimator"])
            search = GridSearchCV(pipeline, spec["param_grid"], cv=cv, scoring="roc_auc")
            n_fits = len(ParameterGrid(spec["param_grid"])) * cv.get_n_splits()
            with tqdm_joblib(
                tqdm(total=n_fits, desc=f"  Tuning {model_name}", unit="fit", leave=False)
            ), joblib.parallel_backend("threading", n_jobs=4):
                search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_

            with tqdm_joblib(
                tqdm(
                    total=cv.get_n_splits(),
                    desc=f"  Cross-validating {model_name}",
                    unit="fold",
                    leave=False,
                )
            ), joblib.parallel_backend("threading", n_jobs=4):
                cv_scores = cross_val_score(
                    best_pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
                )

            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", model_name)
            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            ax.set_title(f"Confusion Matrix: {model_name}")
            fig.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            roc_path = FIGURES_DIR / f"roc_curve_{model_name}.png"
            fig, ax = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title(f"ROC Curve: {model_name}")
            fig.tight_layout()
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path))

            # Cast to float64 so MLflow's schema inference doesn't warn about integer
            # columns being unable to represent missing values at inference time.
            input_example = X_train.head(3).astype("float64")
            mlflow.sklearn.log_model(
                best_pipeline, artifact_path="model", input_example=input_example
            )

            model_path = MODELS_DIR / f"{model_name}.joblib"
            joblib.dump(best_pipeline, model_path)

            results[model_name] = {
                "pipeline": best_pipeline,
                "metrics": metrics,
                "cv_roc_auc_mean": cv_scores.mean(),
                "run_id": mlflow.active_run().info.run_id,
                "model_path": model_path,
            }
            logger.info(
                "%s: test metrics=%s cv_roc_auc=%.4f", model_name, metrics, cv_scores.mean()
            )

    best_name = max(results, key=lambda name: results[name]["metrics"]["roc_auc"])
    best = results[best_name]
    logger.info("Best model: %s (test ROC-AUC=%.4f)", best_name, best["metrics"]["roc_auc"])

    joblib.dump(best["pipeline"], MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    return {"best_model_name": best_name, "results": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    df = load_dataset()
    train_and_track(df)


if __name__ == "__main__":
    main()
