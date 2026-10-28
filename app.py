"""
Flask REST API for the AAVAIL revenue prediction service.

Endpoints
---------
GET  /                      → health check
GET  /predict/<country>     → predictions for a specific country
GET  /predict/all           → predictions for all countries combined
GET  /countries             → list available countries
GET  /metrics               → performance metrics
POST /train                 → trigger full training pipeline
"""

import os
from flask import Flask, jsonify, request
from config import get_config, ensure_directories
from logger import get_logger, log_event
from monitoring import PerformanceMonitor
from model import predict, train_all
from data_ingestion import load_data, clean_data, COUNTRIES


def create_app(cfg=None):
    """Application factory.

    Parameters
    ----------
    cfg : config object, optional
        Pass a TestConfig for testing.

    Returns
    -------
    app : Flask
    """
    if cfg is None:
        cfg = get_config()
    ensure_directories(cfg)

    app = Flask(__name__)
    app.config["TESTING"] = cfg.TESTING
    app.config["ENV_NAME"] = cfg.ENV

    logger = get_logger("api", cfg)
    monitor = PerformanceMonitor(cfg)

    # Store cfg and monitor on the app for test access
    app.app_cfg = cfg
    app.monitor = monitor

    # ── Routes ──────────────────────────────────

    @app.route("/", methods=["GET"])
    def health():
        """Health-check endpoint."""
        log_event(logger, "API", "Health check")
        return jsonify({
            "status": "healthy",
            "env": cfg.ENV,
            "message": "AAVAIL Revenue Prediction API"
        })

    @app.route("/countries", methods=["GET"])
    def list_countries():
        """Return available countries."""
        log_event(logger, "API", "List countries")
        return jsonify({"countries": COUNTRIES})

    @app.route("/predict/<country>", methods=["GET"])
    def get_prediction(country):
        """Return predictions for a specific country or 'all'."""
        with monitor.time(f"predict_{country}"):
            log_event(logger, "API", "Prediction request", country=country)
            result = predict(country, cfg)

        if "error" in result:
            return jsonify(result), 404

        return jsonify(result)

    @app.route("/train", methods=["POST"])
    def trigger_train():
        """Trigger the full training pipeline."""
        with monitor.time("train"):
            log_event(logger, "API", "Training triggered")
            summary = train_all(cfg)

        return jsonify({
            "status": "training complete",
            "summary": {k: v for k, v in summary.items()}
        })

    @app.route("/metrics", methods=["GET"])
    def metrics():
        """Return performance monitoring metrics."""
        return jsonify(monitor.get_metrics())

    return app


# ─── Run directly ───
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=False)
