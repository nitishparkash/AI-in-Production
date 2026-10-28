"""
Model training, comparison, and prediction module.

Compares three regression models:
  1. Ridge Regression  (baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Saves the best model per country (and an "all" model), along with a
baseline-vs-best comparison visualization.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import get_config, ensure_directories
from logger import get_logger, log_event
from data_ingestion import load_data, clean_data, get_feature_matrix

# ──────────────────────────────────────────────
# Model catalogue
# ──────────────────────────────────────────────
MODEL_CATALOGUE = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    ),
}


def _build_pipeline(model):
    """Wrap a model in a scaling pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _evaluate(y_true, y_pred):
    """Return a dict of regression metrics."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_models(df, label, cfg=None):
    """Train and compare all models on the given dataframe slice.

    Parameters
    ----------
    df : pd.DataFrame
        Subset (one country or the entire dataset).
    label : str
        Identifier such as country name or 'all'.
    cfg : config object, optional

    Returns
    -------
    results : dict
        {model_name: {metrics, pipeline, is_best}}
    best_name : str
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("model", cfg)

    X, y = get_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    results = {}
    best_r2 = -np.inf
    best_name = None

    for name, base_model in MODEL_CATALOGUE.items():
        pipe = _build_pipeline(base_model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _evaluate(y_test, y_pred)
        results[name] = {"metrics": metrics, "pipeline": pipe, "is_best": False}
        log_event(logger, "MODEL", f"Trained {name} for '{label}'",
                  **metrics)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name

    results[best_name]["is_best"] = True

    # Persist the best model
    model_path = os.path.join(cfg.MODEL_DIR, f"model_{label}.joblib")
    joblib.dump(results[best_name]["pipeline"], model_path)

    # Save comparison metrics to JSON
    metrics_path = os.path.join(cfg.MODEL_DIR, f"metrics_{label}.json")
    metrics_out = {n: r["metrics"] for n, r in results.items()}
    metrics_out["_best"] = best_name
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    log_event(logger, "MODEL", f"Best model for '{label}': {best_name}",
              r2=best_r2, saved=model_path)

    return results, best_name


# ──────────────────────────────────────────────
# Model comparison visualisation
# ──────────────────────────────────────────────
def plot_model_comparison(results, label, cfg=None):
    """Bar chart comparing R² of all models (highlights baseline vs best).

    Parameters
    ----------
    results : dict
        Output of train_models().
    label : str
    cfg : config object, optional

    Returns
    -------
    path : str
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("model", cfg)

    names = list(results.keys())
    r2_scores = [results[n]["metrics"]["r2"] for n in names]
    colors = ["#2ecc71" if results[n]["is_best"] else "#3498db" for n in names]

    # Mark Ridge as the baseline with a distinct colour
    for i, n in enumerate(names):
        if n == "ridge":
            colors[i] = "#e74c3c"  # red = baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, r2_scores, color=colors, edgecolor="white", linewidth=1.2)
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(f"Model Comparison — {label}", fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score")
    ax.set_ylim(0, max(r2_scores) * 1.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Baseline (Ridge)"),
        Patch(facecolor="#2ecc71", label="Best model"),
        Patch(facecolor="#3498db", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()

    path = os.path.join(cfg.EDA_DIR, f"model_comparison_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "MODEL", "Saved model comparison chart", path=path)
    return path


def plot_prediction_vs_actual(df, label, cfg=None):
    """Scatter: predicted vs actual revenue for the best model.

    Parameters
    ----------
    df : pd.DataFrame
    label : str
    cfg : config object, optional

    Returns
    -------
    path : str
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("model", cfg)

    model_path = os.path.join(cfg.MODEL_DIR, f"model_{label}.joblib")
    pipe = joblib.load(model_path)

    X, y = get_feature_matrix(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    y_pred = pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=10, color="#3498db")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"Predicted vs Actual — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(cfg.EDA_DIR, f"pred_vs_actual_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_event(logger, "MODEL", "Saved pred vs actual chart", path=path)
    return path


# ──────────────────────────────────────────────
# Prediction (inference)
# ──────────────────────────────────────────────
def predict(country, cfg=None):
    """Load a trained model and return predictions on the latest data slice.

    Parameters
    ----------
    country : str
        Country name or 'all'.
    cfg : config object, optional

    Returns
    -------
    dict with prediction results and metadata.
    """
    if cfg is None:
        cfg = get_config()
    logger = get_logger("model", cfg)

    model_path = os.path.join(cfg.MODEL_DIR, f"model_{country}.joblib")
    if not os.path.exists(model_path):
        log_event(logger, "MODEL", f"Model not found for '{country}'")
        return {"error": f"No trained model for '{country}'"}

    pipe = joblib.load(model_path)
    df = clean_data(load_data(cfg), cfg)

    if country != "all":
        df = df[df["country"] == country]
        if df.empty:
            return {"error": f"No data for country '{country}'"}

    # Use the last 30 days for prediction demo
    df_recent = df.sort_values("date").tail(30)
    X, y_actual = get_feature_matrix(df_recent)
    y_pred = pipe.predict(X)
    metrics = _evaluate(y_actual, y_pred)

    log_event(logger, "MODEL", f"Prediction for '{country}'", **metrics)

    return {
        "country": country,
        "n_samples": len(df_recent),
        "predictions": [round(v, 2) for v in y_pred.tolist()],
        "actuals": [round(v, 2) for v in y_actual.tolist()],
        "metrics": metrics,
    }


# ──────────────────────────────────────────────
# Full training pipeline
# ──────────────────────────────────────────────
def train_all(cfg=None):
    """Train models for every country and for 'all'.

    Parameters
    ----------
    cfg : config object, optional

    Returns
    -------
    summary : dict
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)
    logger = get_logger("model", cfg)

    df = clean_data(load_data(cfg), cfg)
    summary = {}

    # Per-country models
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        results, best = train_models(sub, country, cfg)
        plot_model_comparison(results, country, cfg)
        plot_prediction_vs_actual(sub, country, cfg)
        summary[country] = {n: r["metrics"] for n, r in results.items()}
        summary[country]["_best"] = best

    # All-countries model
    results, best = train_models(df, "all", cfg)
    plot_model_comparison(results, "all", cfg)
    plot_prediction_vs_actual(df, "all", cfg)
    summary["all"] = {n: r["metrics"] for n, r in results.items()}
    summary["all"]["_best"] = best

    log_event(logger, "MODEL", "Full training pipeline complete",
              models_trained=(len(df["country"].unique()) + 1) * len(MODEL_CATALOGUE))
    return summary


# ─── CLI ───
if __name__ == "__main__":
    summary = train_all()
    print("\n=== Training Summary ===")
    for label, info in summary.items():
        best = info.get("_best", "?")
        best_r2 = info.get(best, {}).get("r2", "?")
        print(f"  {label:20s}  best={best:20s}  R²={best_r2}")
