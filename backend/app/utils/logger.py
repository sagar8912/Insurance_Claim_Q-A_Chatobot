"""Structured logging utility with API key redaction and clean console formatting."""
import logging
import re
import sys
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts API keys and tokens from logs."""

    PATTERNS = [
        re.compile(r"gsk_[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
        re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([\w\-]+)(['\"]?)", re.IGNORECASE),
        re.compile(r"(bearer\s+)([\w\-\.]+)", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            text = record.msg
            for pattern in self.PATTERNS:
                if "api" in pattern.pattern.lower():
                    text = pattern.sub(r"\1***REDACTED***\3", text)
                elif "bearer" in pattern.pattern.lower():
                    text = pattern.sub(r"\1***REDACTED***", text)
                else:
                    text = pattern.sub("***REDACTED***", text)
            record.msg = text
        return True


def setup_logger(
    name: str = "insurance_rag",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a centralized application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "insurance_rag") -> logging.Logger:
    """Convenience getter for child or module loggers."""
    return logging.getLogger(name)
