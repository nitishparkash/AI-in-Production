"""
Unit tests for the Flask API.

All tests use TestConfig so that production models, data, and logs
are never touched.
"""

import os
import json
import shutil
import unittest

# Ensure test environment is set BEFORE importing app modules
os.environ["APP_ENV"] = "test"

from config import TestConfig, ensure_directories
from app import create_app
from data_ingestion import generate_data, clean_data
from model import train_models


class TestAPI(unittest.TestCase):
    """Test suite for the Flask REST API."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train a test model (isolated from production)."""
        cls.cfg = TestConfig()
        ensure_directories(cls.cfg)

        # Generate test data
        cls.df = generate_data(cls.cfg, n_years=1)
        cls.df = clean_data(cls.df, cls.cfg)

        # Train models for one country and 'all' so predict endpoints work
        country_df = cls.df[cls.df["country"] == "united_kingdom"]
        train_models(country_df, "united_kingdom", cls.cfg)
        train_models(cls.df, "all", cls.cfg)

    @classmethod
    def tearDownClass(cls):
        """Remove temporary test directories."""
        for d in [cls.cfg.DATA_DIR, cls.cfg.MODEL_DIR,
                  cls.cfg.LOG_DIR, cls.cfg.EDA_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)

    def setUp(self):
        """Create a test client."""
        self.app = create_app(self.cfg)
        self.client = self.app.test_client()

    # ── Health check ──
    def test_health_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["env"], "test")

    # ── Countries list ──
    def test_countries_endpoint(self):
        resp = self.client.get("/countries")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("countries", data)
        self.assertIsInstance(data["countries"], list)
        self.assertGreater(len(data["countries"]), 0)

    # ── Predict for a specific country ──
    def test_predict_country(self):
        resp = self.client.get("/predict/united_kingdom")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["country"], "united_kingdom")
        self.assertIn("predictions", data)
        self.assertIn("metrics", data)
        self.assertGreater(len(data["predictions"]), 0)

    # ── Predict for all countries ──
    def test_predict_all(self):
        resp = self.client.get("/predict/all")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["country"], "all")
        self.assertIn("predictions", data)

    # ── Unknown country → 404 ──
    def test_predict_unknown_country(self):
        resp = self.client.get("/predict/atlantis")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)

    # ── Metrics endpoint ──
    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("uptime_seconds", data)
        self.assertIn("system", data)

    # ── Train endpoint ──
    def test_train_endpoint(self):
        resp = self.client.post("/train")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "training complete")
        self.assertIn("summary", data)


if __name__ == "__main__":
    unittest.main()
