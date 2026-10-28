"""
Configuration module for AI-in-Production project.
Separates production and test settings to ensure test isolation.
"""

import os

# ─────────────────────────────────────────────
# Base paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ProductionConfig:
    """Production configuration."""
    ENV = "production"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    EDA_DIR = os.path.join(BASE_DIR, "eda_outputs")
    DATA_FILE = os.path.join(DATA_DIR, "aavail_data.csv")
    LOG_FILE = os.path.join(LOG_DIR, "production.log")
    MONITORING_LOG = os.path.join(LOG_DIR, "monitoring.log")
    DEBUG = False
    TESTING = False


class TestConfig:
    """Test configuration — fully isolated from production."""
    ENV = "test"
    DATA_DIR = os.path.join(BASE_DIR, "tests", "tmp_data")
    MODEL_DIR = os.path.join(BASE_DIR, "tests", "tmp_models")
    LOG_DIR = os.path.join(BASE_DIR, "tests", "tmp_logs")
    EDA_DIR = os.path.join(BASE_DIR, "tests", "tmp_eda")
    DATA_FILE = os.path.join(DATA_DIR, "test_aavail_data.csv")
    LOG_FILE = os.path.join(LOG_DIR, "test.log")
    MONITORING_LOG = os.path.join(LOG_DIR, "test_monitoring.log")
    DEBUG = True
    TESTING = True


def get_config(env=None):
    """Return the appropriate configuration object.

    Parameters
    ----------
    env : str, optional
        'test' for test config, anything else for production.

    Returns
    -------
    config : ProductionConfig or TestConfig
    """
    if env is None:
        env = os.environ.get("APP_ENV", "production")
    if env == "test":
        return TestConfig()
    return ProductionConfig()


def ensure_directories(cfg):
    """Create all required directories for the given configuration."""
    for d in [cfg.DATA_DIR, cfg.MODEL_DIR, cfg.LOG_DIR, cfg.EDA_DIR]:
        os.makedirs(d, exist_ok=True)
