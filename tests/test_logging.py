"""
Unit tests for the logging module.

Verifies that:
  - Log files are created in the test directory (not production).
  - Structured log events contain the expected fields.
  - The function-call decorator records timing info.
  - Test logs are fully isolated from production logs.
"""

import os
import shutil
import logging
import unittest
import time

os.environ["APP_ENV"] = "test"

from config import TestConfig, ProductionConfig, ensure_directories
from logger import get_logger, log_event, log_function_call


class TestLogging(unittest.TestCase):
    """Test suite for the logging module."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = TestConfig()
        ensure_directories(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        # Close all handlers to release file locks before cleanup
        for name in list(logging.Logger.manager.loggerDict):
            lgr = logging.getLogger(name)
            for handler in lgr.handlers[:]:
                handler.close()
                lgr.removeHandler(handler)

        for d in [cls.cfg.DATA_DIR, cls.cfg.MODEL_DIR,
                  cls.cfg.LOG_DIR, cls.cfg.EDA_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)

    # ── Logger is created ──
    def test_logger_creation(self):
        logger = get_logger("test_module", self.cfg)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")

    # ── Log file is written in test directory ──
    def test_log_file_created(self):
        logger = get_logger("test_file_check", self.cfg)
        logger.info("test message")
        # Force flush
        for h in logger.handlers:
            h.flush()
        self.assertTrue(os.path.exists(self.cfg.LOG_FILE))

    # ── Structured event appears in log file ──
    def test_log_event_content(self):
        logger = get_logger("test_event", self.cfg)
        log_event(logger, "TEST", "unit test event", key1="value1")
        for h in logger.handlers:
            h.flush()

        with open(self.cfg.LOG_FILE, "r") as f:
            content = f.read()
        self.assertIn("[TEST]", content)
        self.assertIn("unit test event", content)
        self.assertIn("key1=value1", content)

    # ── Function call decorator logs entry and exit ──
    def test_log_function_call_decorator(self):
        logger = get_logger("test_decorator", self.cfg)

        @log_function_call(logger)
        def sample_func():
            time.sleep(0.05)
            return 42

        result = sample_func()
        self.assertEqual(result, 42)

        for h in logger.handlers:
            h.flush()

        with open(self.cfg.LOG_FILE, "r") as f:
            content = f.read()
        self.assertIn("CALL sample_func", content)
        self.assertIn("done", content)

    # ── Log level is DEBUG ──
    def test_logger_level(self):
        logger = get_logger("test_level", self.cfg)
        self.assertEqual(logger.level, logging.DEBUG)

    # ── Isolation: test log path differs from production ──
    def test_isolation_from_production(self):
        prod_cfg = ProductionConfig()
        self.assertNotEqual(
            os.path.abspath(self.cfg.LOG_FILE),
            os.path.abspath(prod_cfg.LOG_FILE),
        )

    # ── No duplicate handlers when get_logger called twice ──
    def test_no_duplicate_handlers(self):
        name = "test_dup_handlers"
        l1 = get_logger(name, self.cfg)
        n1 = len(l1.handlers)
        l2 = get_logger(name, self.cfg)
        n2 = len(l2.handlers)
        self.assertEqual(n1, n2)


if __name__ == "__main__":
    unittest.main()
