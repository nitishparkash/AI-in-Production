"""
Exploratory Data Analysis (EDA) module.

Generates and saves visualizations for the AAVAIL revenue dataset:
  1. Revenue over time by country
  2. Revenue distribution per country (box plot)
  3. Monthly aggregated revenue heatmap
  4. Correlation heatmap of numeric features
  5. Customer vs Revenue scatter

All figures are saved as PNGs in the configured EDA output directory.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Docker / CI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import get_config, ensure_directories
from logger import get_logger, log_event
from data_ingestion import load_data, clean_data

# ─── Style ───
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def plot_revenue_over_time(df, cfg):
    """Line chart: daily revenue per country."""
    logger = get_logger("eda", cfg)
    fig, ax = plt.subplots(figsize=(14, 6))
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        ax.plot(sub["date"], sub["revenue"], label=country, alpha=0.7, linewidth=0.8)
    ax.set_title("Daily Revenue Over Time by Country")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(cfg.EDA_DIR, "revenue_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "EDA", "Saved revenue_over_time plot", path=path)


def plot_revenue_distribution(df, cfg):
    """Box plot: revenue distribution per country."""
    logger = get_logger("eda", cfg)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x="country", y="revenue", ax=ax)
    ax.set_title("Revenue Distribution by Country")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(cfg.EDA_DIR, "revenue_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "EDA", "Saved revenue_distribution plot", path=path)


def plot_monthly_heatmap(df, cfg):
    """Heatmap: average monthly revenue per country."""
    logger = get_logger("eda", cfg)
    pivot = df.groupby(["country", "month"])["revenue"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Avg Revenue"})
    ax.set_title("Average Monthly Revenue Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Country")
    plt.tight_layout()
    path = os.path.join(cfg.EDA_DIR, "monthly_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "EDA", "Saved monthly_heatmap plot", path=path)


def plot_correlation_matrix(df, cfg):
    """Heatmap: Pearson correlation of numeric features."""
    logger = get_logger("eda", cfg)
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(cfg.EDA_DIR, "correlation_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "EDA", "Saved correlation_matrix plot", path=path)


def plot_customers_vs_revenue(df, cfg):
    """Scatter: number of customers vs revenue coloured by country."""
    logger = get_logger("eda", cfg)
    fig, ax = plt.subplots(figsize=(10, 7))
    for country in df["country"].unique():
        sub = df[df["country"] == country].sample(min(200, len(df)), random_state=42)
        ax.scatter(sub["num_customers"], sub["revenue"], label=country, alpha=0.5, s=12)
    ax.set_title("Customers vs Revenue")
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Revenue")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(cfg.EDA_DIR, "customers_vs_revenue.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "EDA", "Saved customers_vs_revenue plot", path=path)


def run_eda(cfg=None):
    """Execute the full EDA pipeline and save all visualizations.

    Parameters
    ----------
    cfg : config object, optional

    Returns
    -------
    list[str]
        Paths to saved figures.
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("eda", cfg)

    log_event(logger, "EDA", "Starting EDA pipeline")
    df = clean_data(load_data(cfg), cfg)

    plot_revenue_over_time(df, cfg)
    plot_revenue_distribution(df, cfg)
    plot_monthly_heatmap(df, cfg)
    plot_correlation_matrix(df, cfg)
    plot_customers_vs_revenue(df, cfg)

    outputs = [f for f in os.listdir(cfg.EDA_DIR) if f.endswith(".png")]
    log_event(logger, "EDA", "EDA pipeline complete", charts=len(outputs))
    return [os.path.join(cfg.EDA_DIR, f) for f in outputs]


# ─── CLI ───
if __name__ == "__main__":
    paths = run_eda()
    print(f"EDA complete — {len(paths)} charts saved:")
    for p in paths:
        print(f"  → {p}")
