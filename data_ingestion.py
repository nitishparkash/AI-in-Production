"""
Data ingestion module for the AAVAIL revenue prediction project.

Provides functions to:
  1. Generate a synthetic AAVAIL-style revenue dataset.
  2. Load and clean the dataset for downstream modelling.

Designed as importable functions so it can be called from scripts,
notebooks, or the Flask API — facilitating automation.
"""

import os
import numpy as np
import pandas as pd
from config import get_config, ensure_directories
from logger import get_logger, log_event

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COUNTRIES = [
    "united_kingdom", "hong_kong", "portugal", "singapore",
    "france", "germany", "spain", "netherlands",
    "norway", "japan"
]

SEED = 42


def generate_data(cfg=None, n_years=3):
    """Generate a synthetic AAVAIL-style revenue dataset and save to CSV.

    The dataset contains daily revenue data for multiple countries with
    realistic trends, seasonality, and noise.

    Parameters
    ----------
    cfg : config object, optional
    n_years : int
        Number of years of data to generate.

    Returns
    -------
    df : pd.DataFrame
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("data_ingestion", cfg)

    np.random.seed(SEED)
    dates = pd.date_range(
        start="2018-01-01",
        periods=365 * n_years,
        freq="D"
    )

    rows = []
    for country in COUNTRIES:
        base_revenue = np.random.uniform(50_000, 500_000)
        trend = np.linspace(0, np.random.uniform(0.05, 0.25) * base_revenue, len(dates))
        seasonality = base_revenue * 0.15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        noise = np.random.normal(0, base_revenue * 0.05, len(dates))
        revenue = base_revenue + trend + seasonality + noise

        # Derived features
        n_customers = (revenue / np.random.uniform(80, 200)).astype(int)
        n_invoices = (n_customers * np.random.uniform(0.8, 1.5)).astype(int)
        unit_price_mean = revenue / (n_invoices + 1)

        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "country": country,
                "revenue": round(revenue[i], 2),
                "num_customers": int(n_customers[i]),
                "num_invoices": int(n_invoices[i]),
                "avg_unit_price": round(unit_price_mean[i], 2),
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "day_of_week": date.dayofweek,
                "day_of_year": date.dayofyear,
            })

    df = pd.DataFrame(rows)
    df.to_csv(cfg.DATA_FILE, index=False)
    log_event(logger, "DATA", "Dataset generated",
              rows=len(df), countries=len(COUNTRIES), path=cfg.DATA_FILE)
    return df


def load_data(cfg=None):
    """Load and validate the AAVAIL dataset from CSV.

    If the CSV does not exist it is generated automatically.

    Parameters
    ----------
    cfg : config object, optional

    Returns
    -------
    df : pd.DataFrame
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("data_ingestion", cfg)

    if not os.path.exists(cfg.DATA_FILE):
        log_event(logger, "DATA", "Data file not found — generating", path=cfg.DATA_FILE)
        return generate_data(cfg)

    df = pd.read_csv(cfg.DATA_FILE, parse_dates=["date"])
    log_event(logger, "DATA", "Dataset loaded", rows=len(df), columns=len(df.columns))
    return df


def clean_data(df, cfg=None):
    """Clean the dataset: drop nulls, remove negatives, sort.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : config object, optional

    Returns
    -------
    df : pd.DataFrame  (cleaned copy)
    """
    if cfg is None:
        cfg = get_config()
    logger = get_logger("data_ingestion", cfg)

    before = len(df)
    df = df.dropna().copy()
    df = df[df["revenue"] > 0]
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    after = len(df)
    log_event(logger, "DATA", "Dataset cleaned",
              rows_before=before, rows_after=after, dropped=before - after)
    return df


def get_country_data(df, country):
    """Filter dataframe for a specific country.

    Parameters
    ----------
    df : pd.DataFrame
    country : str

    Returns
    -------
    pd.DataFrame
    """
    return df[df["country"] == country].copy()


def get_feature_matrix(df):
    """Build feature matrix X and target vector y for modelling.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    feature_cols = [
        "num_customers", "num_invoices", "avg_unit_price",
        "year", "month", "day", "day_of_week", "day_of_year"
    ]
    X = df[feature_cols].copy()
    y = df["revenue"].copy()
    return X, y


# ──────────────────────────────────────────────
# CLI entry point for automation
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Running data ingestion pipeline …")
    cfg = get_config()
    df = generate_data(cfg)
    df = clean_data(df, cfg)
    print(f"Done — {len(df)} rows written to {cfg.DATA_FILE}")
