"""Tests for structured logging configuration."""

import logging

from osmose.logging import setup_logging


def test_setup_logging_returns_logger():
    logger = setup_logging("test_osmose")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_osmose"


def test_setup_logging_default_level():
    logger = setup_logging("test_default")
    assert logger.level == logging.INFO


def test_setup_logging_custom_level():
    logger = setup_logging("test_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_logger_has_handler():
    logger = setup_logging("test_handler")
    assert len(logger.handlers) >= 1


def test_handler_is_stream_handler():
    logger = setup_logging("test_stream")
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 1


def test_no_duplicate_handlers():
    """Calling setup_logging twice with same name should not add duplicate handlers."""
    name = "test_no_dup"
    logger1 = setup_logging(name)
    handler_count = len(logger1.handlers)
    logger2 = setup_logging(name)
    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count


def test_formatter_format():
    logger = setup_logging("test_fmt")
    handler = logger.handlers[0]
    assert "%(asctime)s" in handler.formatter._fmt
    assert "%(name)s" in handler.formatter._fmt
    assert "%(levelname)s" in handler.formatter._fmt
