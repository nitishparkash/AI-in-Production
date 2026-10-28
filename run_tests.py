#!/usr/bin/env python
"""
Single script to discover and run ALL unit tests.

Usage:
    python run_tests.py
"""

import os
import sys
import unittest

# Force test environment so tests never touch production artefacts
os.environ["APP_ENV"] = "test"


def main():
    """Discover and run all tests in the tests/ directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), "tests"),
        pattern="test_*.py",
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if any test failed
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
