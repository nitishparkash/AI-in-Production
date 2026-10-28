"""
Unit tests for the model module.

Uses TestConfig to ensure all model artefacts are written to isolated
temporary directories — never touching production models.
"""

import os
import shutil
import unittest
import joblib
import numpy as np

os.environ["APP_ENV"] = "test"

from config import TestConfig, ensure_directories
from data_ingestion import generate_data, clean_data, get_feature_matrix
from model import train_models, predict, train_all, MODEL_CATALOGUE


class TestModel(unittest.TestCase):
    """Test suite for model training and prediction."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = TestConfig()
        ensure_directories(cls.cfg)
        cls.df = generate_data(cls.cfg, n_years=1)
        cls.df = clean_data(cls.df, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        for d in [cls.cfg.DATA_DIR, cls.cfg.MODEL_DIR,
                  cls.cfg.LOG_DIR, cls.cfg.EDA_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)

    # ── Feature extraction ──
    def test_feature_matrix_shape(self):
        X, y = get_feature_matrix(self.df)
        self.assertEqual(len(X), len(y))
        self.assertGreater(X.shape[1], 0)

    # ── Multiple models are trained ──
    def test_multiple_models_compared(self):
        sub = self.df[self.df["country"] == "hong_kong"]
        results, best = train_models(sub, "hong_kong", self.cfg)
        self.assertEqual(len(results), len(MODEL_CATALOGUE))
        self.assertIn(best, MODEL_CATALOGUE)

    # ── Best model is saved ──
    def test_model_saved(self):
        sub = self.df[self.df["country"] == "portugal"]
        train_models(sub, "portugal", self.cfg)
        path = os.path.join(self.cfg.MODEL_DIR, "model_portugal.joblib")
        self.assertTrue(os.path.exists(path))

    # ── Loaded model can predict ──
    def test_model_predict(self):
        sub = self.df[self.df["country"] == "portugal"]
        train_models(sub, "portugal", self.cfg)
        result = predict("portugal", self.cfg)
        self.assertNotIn("error", result)
        self.assertIn("predictions", result)
        self.assertGreater(len(result["predictions"]), 0)

    # ── Metrics contain expected keys ──
    def test_metrics_keys(self):
        sub = self.df[self.df["country"] == "singapore"]
        results, _ = train_models(sub, "singapore", self.cfg)
        for name, info in results.items():
            self.assertIn("mae", info["metrics"])
            self.assertIn("rmse", info["metrics"])
            self.assertIn("r2", info["metrics"])

    # ── R² is reasonable (> 0 on synthetic data) ──
    def test_r2_positive(self):
        sub = self.df[self.df["country"] == "france"]
        results, best = train_models(sub, "france", self.cfg)
        best_r2 = results[best]["metrics"]["r2"]
        self.assertGreater(best_r2, 0.0)

    # ── train_all trains models for every country + 'all' ──
    def test_train_all(self):
        summary = train_all(self.cfg)
        self.assertIn("all", summary)
        countries = self.df["country"].unique()
        for c in countries:
            self.assertIn(c, summary)

    # ── Predict for unknown country returns error ──
    def test_predict_unknown(self):
        result = predict("atlantis", self.cfg)
        self.assertIn("error", result)

    # ── Isolation: test models are NOT in production dir ──
    def test_isolation_from_production(self):
        from config import ProductionConfig
        prod = ProductionConfig()
        test_model = os.path.join(self.cfg.MODEL_DIR, "model_hong_kong.joblib")
        prod_model = os.path.join(prod.MODEL_DIR, "model_hong_kong.joblib")
        if os.path.exists(test_model):
            self.assertNotEqual(
                os.path.abspath(test_model),
                os.path.abspath(prod_model)
            )


if __name__ == "__main__":
    unittest.main()
