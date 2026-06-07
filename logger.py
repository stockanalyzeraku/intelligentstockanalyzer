# =============================================================================
# CELL 4 — Logger
# =============================================================================
"""
Structured JSON logger. Every module creates its own StructuredLogger instance.
Writes to a daily rotating log file on Drive and prints WARNING+ to Colab output.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import config

class StructuredLogger:
    """
    Structured JSON logger for the Investment Brain Agent.

    Writes every log entry as a single-line JSON object to a daily log file.
    Entries at WARNING level and above are also echoed to stdout so they
    appear in Colab cell output.

    Parameters
    ----------
    component : str
        Name of the module/component creating this logger (e.g. "agent").
    config : Config
        Project config instance for resolving log paths.
    """

    def __init__(self, component: str, config=None):
        """Initialise logger for the given component."""
        self._component = component
        self._config = config
        self._ist = timezone(timedelta(hours = 5, minutes = 30))
        self._log_file = self._resolve_log_path()
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a DEBUG-level message."""
        self._write("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an INFO-level message."""
        self._write("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a WARNING-level message and echo to stdout."""
        self._write("WARNING", message, echo=True, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an ERROR-level message and echo to stdout."""
        self._write("ERROR", message, echo=True, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a CRITICAL-level message and echo to stdout."""
        self._write("CRITICAL", message, echo=True, **kwargs)

    # ── Private helpers ───────────────────────────────────────────────────

    def _resolve_log_path(self) -> str:
        """Compute today's log file path."""
        date_str = datetime.now(self._ist).strftime("%Y-%m-%d")
        filename = f"{date_str}_{self._component}.log"
        return os.path.join(self._config.LOGS_PATH, filename)

    def _write(self, level: str, message: str, echo: bool = False, **kwargs: Any) -> None:
        """
        Serialise and write one log entry.

        Parameters
        ----------
        level : str
            Log level string.
        message : str
            Human-readable log message.
        echo : bool
            Whether to also print to stdout.
        **kwargs
            Arbitrary extra fields included in the JSON entry.
        """
        entry: Dict[str, Any] = {
            "ts": datetime.now(self._ist).isoformat(),
            "level": level,
            "component": self._component,
            "message": message,
        }
        entry.update(kwargs)
        line = json.dumps(entry, default=str)
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            # Last-resort: if we cannot write to disk, print the entry
            print(f"[Logger write failure] {exc} | entry={line}", file=sys.stderr)
        if echo:
            print(f"[{level}] {self._component}: {message}", file=sys.stderr)


def get_logger(component: str) -> StructuredLogger:
    """
    Factory function to obtain a StructuredLogger for a named component.

    Parameters
    ----------
    component : str
        Component name used in log entries and log filename.

    Returns
    -------
    StructuredLogger
    """
    
    return StructuredLogger(component, config.Config())


# Module-level logger for Cell 4 itself
_logger = get_logger("logger")
_logger.info("Logger initialised.")

# ----------------------------------------------------------------------------
# Cell 4: Logger
# Purpose: Provide structured JSON logging to file + WARNING-echo to stdout.
# Key Classes: StructuredLogger
# Key Functions: get_logger(component) → StructuredLogger,
#   StructuredLogger.debug/info/warning/error/critical(message, **kwargs) → None
# Key Constants/Config: CONFIG.LOGS_PATH
# Imports exported: StructuredLogger, get_logger
# Depends on: Cell 3 (CONFIG)
# Critical notes: Never use print() for operational messages — use get_logger().
#   Each module should call get_logger(__name__ or component string).
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
