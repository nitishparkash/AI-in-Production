"""
Centralized logging module for AI-in-Production project.
Provides structured logging with separate handlers for production and testing.
"""

import os
import logging
import time
from functools import wraps
from config import get_config, ensure_directories


def get_logger(name, cfg=None):
    """Create and return a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    cfg : config object, optional
        Configuration; defaults to current environment config.

    Returns
    -------
    logger : logging.Logger
    """
    if cfg is None:
        cfg = get_config()

    ensure_directories(cfg)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── File handler ──
    fh = logging.FileHandler(cfg.LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # ── Console handler ──
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def log_event(logger, event_type, message, **kwargs):
    """Log a structured event.

    Parameters
    ----------
    logger : logging.Logger
    event_type : str
        Category such as 'DATA', 'MODEL', 'API', 'MONITOR'.
    message : str
    **kwargs
        Extra key-value pairs appended to the log line.
    """
    extras = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    full_message = f"[{event_type}] {message}"
    if extras:
        full_message += f" | {extras}"
    logger.info(full_message)


def log_function_call(logger):
    """Decorator that logs function entry, exit, and duration.

    Parameters
    ----------
    logger : logging.Logger
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"CALL {func.__name__} — start")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = round(time.time() - start, 4)
            logger.info(f"CALL {func.__name__} — done ({elapsed}s)")
            return result
        return wrapper
    return decorator
