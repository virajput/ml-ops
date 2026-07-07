"""Generate exploratory data analysis plots for the cleaned Heart Disease dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_disease.data import (  # noqa: E402
    NUMERIC_FEATURES,
    PROCESSED_PATH,
    TARGET,
    get_clean_dataset,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index.map({0: "No Disease", 1: "Disease"}), y=counts.values, ax=ax)
    ax.set_title("Class Balance: Heart Disease Presence")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, palette="Set1", alpha=0.6)
        ax.set_title(f"Distribution of {col}")
    for ax in axes.flat[len(NUMERIC_FEATURES):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_histograms.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df.isna().sum()
    sns.barplot(x=missing.index, y=missing.values, ax=ax)
    ax.set_title("Missing Values per Column (post-cleaning)")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close(fig)


def plot_feature_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET, y="thalach", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_title("Max Heart Rate Achieved vs. Disease Presence")
    fig.tight_layout()
    fig.savefig(out_dir / "thalach_vs_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="cp", hue=TARGET, ax=ax)
    ax.set_title("Chest Pain Type vs. Disease Presence")
    fig.tight_layout()
    fig.savefig(out_dir / "chest_pain_vs_target.png", dpi=150)
    plt.close(fig)


def run_eda(processed_path: Path = PROCESSED_PATH, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not processed_path.exists():
        get_clean_dataset()
    df = pd.read_csv(processed_path)

    plot_class_balance(df, out_dir)
    plot_histograms(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_missing_values(df, out_dir)
    plot_feature_relationships(df, out_dir)

    logger.info("Saved EDA figures to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_eda()
