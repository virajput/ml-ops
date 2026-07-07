"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Columns that are categorical/binary vs. continuous, per the UCI data dictionary.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "heart_disease_raw.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "heart_disease_clean.csv"


def download_raw(dest: Path = RAW_PATH, url: str = DATA_URL) -> Path:
    """Download the raw Cleveland heart-disease CSV from the UCI repository."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_text(response.text)
    logger.info("Saved raw dataset to %s", dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw (headerless) UCI CSV, treating '?' as missing."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe: impute missing values, binarize target, cast dtypes."""
    df = df.copy()

    # `ca` and `thal` contain a handful of '?' values in the raw UCI file. Both are
    # discrete/categorical, so impute with the mode rather than the mean/median.
    for col in ["ca", "thal"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(int)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)

    # Original target `num` is severity 0-4; binarize to presence/absence of disease.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    df = df.reset_index(drop=True)
    return df


def get_clean_dataset(
    raw_path: Path = RAW_PATH,
    processed_path: Path = PROCESSED_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Ensure raw data exists, clean it, persist and return the processed dataframe."""
    if force_download or not raw_path.exists():
        download_raw(raw_path)
    raw_df = load_raw(raw_path)
    clean_df = clean_data(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)
    return clean_df


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    df = get_clean_dataset(force_download=True)
    logger.info("Processed dataset shape: %s", df.shape)
    logger.info("Missing values after cleaning:\n%s", df.isna().sum().to_string())
    logger.info("Class balance:\n%s", df[TARGET].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()
