# =============================================================================
# INVESTMENT BRAIN AGENT — Phase 1 (Google Colab)
# =============================================================================


# =============================================================================
# Install All Dependencies
# =============================================================================
"""
Install all required packages for the Investment Brain Agent.
Run this cell once at the start of each Colab session.
"""

import subprocess
import sys
import re # Added for extracting package names from version specifiers

def install_dependencies():
    """Install all required Python packages for the project."""
    packages = [
        "google-generativeai>=0.7.0",
        "sentence-transformers>=2.7.0",
        "chromadb>=0.5.0",
        "pymupdf>=1.24.0",          # fitz — PDF extraction
        "pdfplumber>=0.11.0",       # table extraction
        "rank_bm25>=0.2.2",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "tenacity>=8.3.0",
    ]

    for pkg_spec in packages:
        # Extract the base package name (e.g., 'google-generativeai' from 'google-generativeai>=0.7.0')
        pkg_name = re.split(r'[<=>~]', pkg_spec)[0].strip()
        try:
            # Install the package silently
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec, "-q"])

            # If successful, get the installed version
            try:
                show_output = subprocess.check_output(
                    [sys.executable, "-m", "pip", "show", pkg_name],
                    stderr=subprocess.STDOUT
                ).decode()
                version_match = re.search(r'^Version: (.+)$', show_output, re.MULTILINE)
                if version_match:
                    version = version_match.group(1)
                    print(f"Package '{pkg_name}' installed successfully (version {version}).")
                else:
                    print(f"Package '{pkg_name}' installed successfully, but installed version could not be determined.")
            except subprocess.CalledProcessError:
                print(f"Package '{pkg_name}' installed successfully, but 'pip show' failed to retrieve version.")

        except subprocess.CalledProcessError as e:
            # If installation fails, print the error output from pip
            error_message = e.output.decode().strip() if e.output else "Unknown error during installation."
            print(f"Error installing package '{pkg_spec}': {error_message}")
        except Exception as e:
            # Catch any other unexpected errors during the process
            print(f"An unexpected error occurred for package '{pkg_spec}': {e}")

install_dependencies()

# ----------------------------------------------------------------------------
# File Name: packages.py
# Purpose: Install every third-party library the project needs.
# Key Classes: None
# Key Functions: install_dependencies() → None
# Key Constants/Config: packages list (hardcoded here only — bootstrapping concern)
# Imports exported: None
# Depends on: None
# Critical notes: Run once per session. Order does not matter.

# =============================================================================
# Mount Drive + Create Folder Structure
# =============================================================================
"""
Mount Google Drive and create the canonical folder structure under /brain/.
Creates all required directories if they do not already exist.
"""

import os

BRAIN_BASE = os.path.dirname(os.path.abspath(__file__))
#BRAIN_BASE = "/workspaces/brain/"
def get_base_path() -> str:
  """Returns Base Path Address"""
  return BRAIN_BASE

def mount_drive():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        print("Google Drive mounted.")
    except ImportError:
        print("[Cell 2] Not running in Colab — skipping Drive mount.")

def create_folder_structure(base_path: str = "/content/drive/MyDrive/brain") -> None:
    """
    Create the full folder hierarchy under base_path.

    Parameters
    ----------
    base_path : str
        Root path for all project files.
    """
    folders = [
        base_path,
        os.path.join(base_path, "uploads"),
        os.path.join(base_path, "chroma_db"),
        os.path.join(base_path, "database"),
        os.path.join(base_path, "logs"),
    ]
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created/Verified: {folder}")
        except PermissionError as e:
            print(f"❌ Permission denied - cannot create folder: {folder}\n   Error: {e}")
        except OSError as e:
            print(f"❌ OS Error - failed to create folder: {folder}\n   Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error creating folder: {folder}\n   Error: {e}")
    print(f"📁 Folder structure verified under: {base_path}")

#mount_drive()

#create_folder_structure(BRAIN_BASE)

# ----------------------------------------------------------------------------
# File Name: folderstructure.py
# Purpose: Mount Google Drive and ensure all project directories exist.
# Key Classes: None
# Key Functions: mount_drive() → None, create_folder_structure(base_path) → None
# Key Constants/Config: BRAIN_BASE
# Imports exported: BRAIN_BASE (used by Cell 3 config)
# Depends on: None
# Critical notes: BRAIN_BASE must match Config.BASE_PATH in Cell 3.
#   In non-Colab environments the Drive mount is skipped gracefully.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------


# =============================================================================
# CELL 3 — Config
# =============================================================================
"""
Central configuration. Single source of truth for all settings.
Every other module imports from this cell — nothing is hardcoded elsewhere.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
#from google.colab import userdata
from folderstructure import get_base_path
from dotenv import load_dotenv

@dataclass
class Config:
    """
    Central configuration class.

    All paths, model names, chunk sizes, timeouts, retry counts,
    and circuit-breaker thresholds live here.
    Call Config.validate() at startup to fail fast on missing prerequisites.
    """

    _instance: Optional["Config"] = None
    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_PATH: str = get_base_path()
    UPLOADS_PATH: str = field(init=False)
    CHROMA_PATH: str = field(init=False)
    DB_PATH: str = field(init=False)
    LOGS_PATH: str = field(init=False)

    # ── LLM ───────────────────────────────────────────────────────────────
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_OUTPUT_TOKENS: int = 4096
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── ChromaDB Collections ──────────────────────────────────────────────
    COL_CHILD: str = "child_chunks"
    COL_PARENT: str = "parent_sections"
    COL_FACTS: str = "financial_facts"
    COL_MGMT: str = "mgmt_statements"

    # ── Chunking ──────────────────────────────────────────────────────────
    PARENT_CHUNK_TOKENS: int = 2500
    CHILD_CHUNK_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────
    SEMANTIC_TOP_K: int = 10
    BM25_TOP_K: int = 10
    FINAL_TOP_K: int = 8
    RRF_K: int = 60
    BM25_MAX_DOCS: int = 100_000

    # ── Agent ─────────────────────────────────────────────────────────────
    MAX_REACT_ITERATIONS: int = 4
    CONVERSATION_HISTORY_LIMIT: int = 5

    # ── Retry / Circuit Breaker ────────────────────────────────────────────
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 30.0
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT: int = 60

    # ── Validation ────────────────────────────────────────────────────────
    MAX_QUESTION_LENGTH: int = 1000
    MIN_QUESTION_LENGTH: int = 5
    MAX_PDF_SIZE_MB: int = 200
    CHUNK_COUNT_MIN: int = 10
    CHUNK_COUNT_MAX: int = 50_000

    # ── SQLite ────────────────────────────────────────────────────────────
    DB_BATCH_SIZE: int = 50

    def __post_init__(self):
        """Derive all path fields from BASE_PATH after object creation."""
        self.UPLOADS_PATH = os.path.join(self.BASE_PATH, "uploads")
        self.CHROMA_PATH = os.path.join(self.BASE_PATH, "chroma_db")
        self.DB_PATH = os.path.join(self.BASE_PATH, "database", "brain.db")
        self.LOGS_PATH = os.path.join(self.BASE_PATH, "logs")

    def validate(self) -> None:
        """
        Validate critical prerequisites at startup.

        Raises
        ------
        EnvironmentError
            If GEMINI_API_KEY is missing or BASE_PATH is not accessible.
        """
        if not self.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Run: import os; os.environ['GEMINI_API_KEY'] = 'your-key'"
            )
        if not os.path.isdir(self.BASE_PATH):
            raise EnvironmentError(
                f"BASE_PATH does not exist: {self.BASE_PATH}. "
                "Run Cell 2 first to create the folder structure."
            )
        print("[Config] Validation passed.")
    
    @classmethod
    def get_instance(cls) -> "Config":
        """
        Return the singleton Config, loading the file if needed.

        Returns
        -------
        Config
        """ 
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance




# Singleton instance used by all modules
CONFIG = Config()

# ----------------------------------------------------------------------------
# File Name: Config
# Purpose: Provide a single Config dataclass as the project's source of truth.
# Key Classes: Config
# Key Functions: Config.validate() → None
# Key Constants/Config: CONFIG (singleton), all Config fields
# Imports exported: CONFIG, Config
# Depends on: None
# Critical notes: CONFIG.validate() must be called before first agent use.
#   All other cells import CONFIG — never re-instantiate Config elsewhere.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

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

# =============================================================================
# CELL 5 — Error Handler
# =============================================================================
"""
Error classification, retry-with-backoff decorator, and CircuitBreaker.
All external calls (Gemini, ChromaDB) must be wrapped with these utilities.
"""

import functools
import random
import threading
import time
from enum import Enum
from typing import Callable, Optional, Type, Tuple
from config import Config
from logger import get_logger


class ErrorType(Enum):
    """Classification of errors by recoverability."""
    TRANSIENT = "transient"   # Retry is worthwhile (rate limit, timeout, network)
    PERMANENT = "permanent"   # Do not retry (bad input, auth failure, not found)


class CircuitBreakerState(Enum):
    """States of the circuit breaker finite-state machine."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing; reject calls immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the circuit breaker is OPEN."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementing CLOSED → OPEN → HALF_OPEN transitions.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "gemini", "chromadb").
    failure_threshold : int
        Number of consecutive failures before opening the circuit.
    recovery_timeout : int
        Seconds to wait before transitioning from OPEN to HALF_OPEN.
    """

    def __init__(self, name: str, failure_threshold: int, recovery_timeout: int):
        """Initialise the circuit breaker in CLOSED state."""
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        self._CONFIG = Config()
        self._logger = get_logger("circuit_breaker")

    @property
    def state(self) -> CircuitBreakerState:
        """Return current state, auto-transitioning OPEN→HALF_OPEN on timeout."""
        with self._lock:
            if (
                self._state == CircuitBreakerState.OPEN
                and self._last_failure_time is not None
                and time.monotonic() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
                self._logger.info(
                    "Circuit breaker transitioned to HALF_OPEN",
                    breaker=self._name,
                )
            return self._state

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute func if circuit allows; raise CircuitBreakerOpenError otherwise.

        Parameters
        ----------
        func : Callable
            The function to execute.
        *args, **kwargs
            Arguments forwarded to func.

        Returns
        -------
        Any
            Return value of func on success.

        Raises
        ------
        CircuitBreakerOpenError
            When the circuit is OPEN.
        Exception
            Re-raises any exception from func after recording failure.
        """
        current = self.state
        if current == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self._name}' is OPEN — call rejected."
            )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _on_success(self) -> None:
        """Reset failure count and close circuit on success."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._logger.info(
                    "Circuit breaker recovered to CLOSED", breaker=self._name
                )

    def _on_failure(self, exc: Exception) -> None:
        """Increment failure count and open circuit if threshold reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._logger.error(
                "Circuit breaker recorded failure",
                breaker=self._name,
                failure_count=self._failure_count,
                error=str(exc),
            )
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._logger.error(
                    "Circuit breaker OPENED",
                    breaker=self._name,
                    failure_count=self._failure_count,
                )


def classify_error(exc: Exception) -> ErrorType:
    """
    Classify an exception as TRANSIENT or PERMANENT.

    Parameters
    ----------
    exc : Exception
        The exception to classify.

    Returns
    -------
    ErrorType
    """
    transient_types = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    transient_messages = (
        "rate limit",
        "quota",
        "503",
        "502",
        "500",
        "429",
        "timeout",
        "temporarily",
        "unavailable",
    )
    if isinstance(exc, transient_types):
        return ErrorType.TRANSIENT
    msg = str(exc).lower()
    if any(kw in msg for kw in transient_messages):
        return ErrorType.TRANSIENT
    return ErrorType.PERMANENT


def retry(
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator: retry with exponential backoff and jitter on TRANSIENT errors.

    Parameters
    ----------
    max_attempts : int, optional
        Override self._CONFIG.RETRY_MAX_ATTEMPTS.
    base_delay : float, optional
        Override self._CONFIG.RETRY_BASE_DELAY.
    max_delay : float, optional
        Override self._CONFIG.RETRY_MAX_DELAY.
    retryable_exceptions : tuple
        Exception types eligible for retry.
    """
    _logger = get_logger("retry")

    def decorator(func: Callable) -> Callable:
        """Wrap func with retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute func, retrying on transient errors."""
            attempts = max_attempts or self._CONFIG.RETRY_MAX_ATTEMPTS
            b_delay = base_delay or self._CONFIG.RETRY_BASE_DELAY
            m_delay = max_delay or self._CONFIG.RETRY_MAX_DELAY
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if classify_error(exc) == ErrorType.PERMANENT:
                        _logger.error(
                            "Permanent error — not retrying",
                            func=func.__name__,
                            error=str(exc),
                        )
                        raise
                    if attempt == attempts:
                        break
                    delay = min(b_delay * (2 ** (attempt - 1)), m_delay)
                    delay += random.uniform(0, delay * 0.1)
                    _logger.warning(
                        "Transient error — retrying",
                        func=func.__name__,
                        attempt=attempt,
                        delay_s=round(delay, 2),
                        error=str(exc),
                    )
                    time.sleep(delay)
            _logger.error(
                "All retry attempts exhausted",
                func=func.__name__,
                attempts=attempts,
                error=str(last_exc),
            )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# Pre-instantiated circuit breakers for shared services
CB_GEMINI = CircuitBreaker(
    "gemini",
    Config.CB_FAILURE_THRESHOLD,
    Config.CB_RECOVERY_TIMEOUT,
)
CB_CHROMADB = CircuitBreaker(
    "chromadb",
    Config.CB_FAILURE_THRESHOLD,
    Config.CB_RECOVERY_TIMEOUT,
)

_eh_logger = get_logger("error_handler")
_eh_logger.info("Error handler initialised.", gemini_cb=CB_GEMINI._name, chroma_cb=CB_CHROMADB._name)

# ----------------------------------------------------------------------------
# Cell 5: Error Handler
# Purpose: Provide ErrorType enum, retry decorator, and CircuitBreaker.
# Key Classes: CircuitBreaker, CircuitBreakerOpenError, ErrorType, CircuitBreakerState
# Key Functions: classify_error(exc) → ErrorType, retry(...) → Callable
# Key Constants/Config: CB_GEMINI, CB_CHROMADB (shared breaker instances)
# Imports exported: CircuitBreaker, CircuitBreakerOpenError, ErrorType,
#   classify_error, retry, CB_GEMINI, CB_CHROMADB
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: CB_GEMINI and CB_CHROMADB are module-level singletons —
#   import them rather than creating new instances.
#   retry() must sit INSIDE the circuit-breaker call, not outside.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

# =============================================================================
# CELL 6 — Input Validator
# =============================================================================
"""
InputValidator: static methods for validating every external input
before it touches the processing pipeline.
"""

import os
import re
from typing import Optional
from logger import get_logger

class ValidationError(ValueError):
    """Raised when input validation fails."""


class InputValidator:
    """
    Static validation methods for all pipeline inputs.

    All methods return the (possibly normalised) input on success
    and raise ValidationError on failure.
    """

    _INJECTION_PATTERNS = re.compile(
        r"(ignore previous|ignore all|disregard|system prompt|"
        r"you are now|pretend you|act as|jailbreak)",
        re.IGNORECASE,
    )

    @staticmethod
    def validate_question(question: str) -> str:
        """
        Validate and sanitise a user question.

        Parameters
        ----------
        question : str
            Raw question from the user.

        Returns
        -------
        str
            Stripped question.

        Raises
        ------
        ValidationError
            If question is empty, too short, too long, or contains
            prompt-injection patterns.
        """
        _v_logger = get_logger("validator")
        if not isinstance(question, str):
            _v_logger.error("Question is not a string", type=type(question).__name__)
            raise ValidationError("Question must be a string.")
        question = question.strip()
        if len(question) < CONFIG.MIN_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too short (min {CONFIG.MIN_QUESTION_LENGTH} chars)."
            )
        if len(question) > CONFIG.MAX_QUESTION_LENGTH:
            raise ValidationError(
                f"Question too long (max {CONFIG.MAX_QUESTION_LENGTH} chars)."
            )
        if InputValidator._INJECTION_PATTERNS.search(question):
            _v_logger.warning("Possible prompt injection detected", question=question[:80])
            raise ValidationError("Question contains disallowed patterns.")
        return question

    @staticmethod
    def validate_scrip(scrip: str) -> str:
        """
        Validate a stock scrip symbol (e.g. 'RELIANCE', 'TCS').

        Parameters
        ----------
        scrip : str
            Scrip symbol to validate.

        Returns
        -------
        str
            Upper-cased scrip.

        Raises
        ------
        ValidationError
            If scrip format is invalid.
        """
        if not isinstance(scrip, str):
            raise ValidationError("Scrip must be a string.")
        scrip = scrip.strip().upper()
        if not re.match(r"^[A-Z0-9&\-]{1,20}$", scrip):
            raise ValidationError(
                f"Invalid scrip format: '{scrip}'. "
                "Expected 1-20 alphanumeric/&/- characters."
            )
        return scrip

    @staticmethod
    def validate_fiscal_year(fy: str) -> str:
        """
        Normalise a fiscal year string to FY25 format.

        Accepts: 'FY2025', '2025', 'fy25', 'FY25', '25'.

        Parameters
        ----------
        fy : str
            Fiscal year in any accepted format.

        Returns
        -------
        str
            Normalised as 'FY25' (two-digit year suffix).

        Raises
        ------
        ValidationError
            If the string cannot be interpreted as a fiscal year.
        """
        if not isinstance(fy, str):
            raise ValidationError("Fiscal year must be a string.")
        fy = fy.strip().upper()
        # FY2025 or 2025
        m = re.match(r"^(?:FY)?(\d{4})$", fy)
        if m:
            return f"FY{m.group(1)[-2:]}"
        # FY25 or 25
        m = re.match(r"^(?:FY)?(\d{2})$", fy)
        if m:
            return f"FY{m.group(1)}"
        raise ValidationError(f"Cannot parse fiscal year from: '{fy}'")

    @staticmethod
    def validate_pdf_path(path: str) -> str:
        """
        Validate a PDF file path for safety and accessibility.

        Parameters
        ----------
        path : str
            Absolute or relative path to a PDF file.

        Returns
        -------
        str
            Resolved absolute path.

        Raises
        ------
        ValidationError
            If path traversal detected, extension wrong, file missing,
            or file exceeds MAX_PDF_SIZE_MB.
        """
        _v_logger = get_logger("validator")
        if not isinstance(path, str):
            raise ValidationError("PDF path must be a string.")
        abs_path = os.path.realpath(path)
        # Path traversal check
        uploads_real = os.path.realpath(CONFIG.UPLOADS_PATH)
        if not abs_path.startswith(uploads_real):
            _v_logger.warning("Path traversal attempt", path=path)
            raise ValidationError(
                f"Path '{path}' is outside the allowed uploads directory."
            )
        if not abs_path.lower().endswith(".pdf"):
            raise ValidationError(f"File must have a .pdf extension: '{path}'")
        if not os.path.isfile(abs_path):
            raise ValidationError(f"File not found: '{abs_path}'")
        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        if size_mb > CONFIG.MAX_PDF_SIZE_MB:
            raise ValidationError(
                f"PDF too large ({size_mb:.1f} MB). Max allowed: {CONFIG.MAX_PDF_SIZE_MB} MB."
            )
        return abs_path

    @staticmethod
    def validate_chunk_count(count: int, context: str = "") -> None:
        """
        Warn if chunk count is outside the expected range.

        Parameters
        ----------
        count : int
            Number of chunks produced.
        context : str
            Description of what was chunked (for log messages).
        """
        _v_logger = get_logger("validator")
        if count < CONFIG.CHUNK_COUNT_MIN:
            _v_logger.warning(
                "Chunk count below expected minimum",
                count=count,
                min=CONFIG.CHUNK_COUNT_MIN,
                context=context,
            )
        elif count > CONFIG.CHUNK_COUNT_MAX:
            _v_logger.warning(
                "Chunk count above expected maximum",
                count=count,
                max=CONFIG.CHUNK_COUNT_MAX,
                context=context,
            )


_val_logger = get_logger("validator")
_val_logger.info("InputValidator ready.")

# ----------------------------------------------------------------------------
# Cell 6: Input Validator
# Purpose: Validate and sanitise all pipeline inputs before processing.
# Key Classes: InputValidator, ValidationError
# Key Functions:
#   InputValidator.validate_question(question) → str
#   InputValidator.validate_scrip(scrip) → str
#   InputValidator.validate_fiscal_year(fy) → str
#   InputValidator.validate_pdf_path(path) → str
#   InputValidator.validate_chunk_count(count, context) → None
# Key Constants/Config: CONFIG.MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH,
#   MAX_PDF_SIZE_MB, CHUNK_COUNT_MIN/MAX, UPLOADS_PATH
# Imports exported: InputValidator, ValidationError
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: validate_pdf_path does a real-path check against
#   CONFIG.UPLOADS_PATH — files outside that tree are rejected.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

# =============================================================================
# CELL 7 — Embedding Model
# =============================================================================
"""
LocalEmbedder: wraps all-MiniLM-L6-v2 via sentence-transformers.
Implements the ChromaDB EmbeddingFunction interface.
No API calls, no quota concerns. Singleton pattern.
"""

from typing import List, Optional
import numpy as np
from logger import get_logger
from config import Config

class LocalEmbedder:
    """
    Local embedding model wrapping all-MiniLM-L6-v2.

    Implements ChromaDB's EmbeddingFunction protocol so it can be passed
    directly to chromadb.Collection constructors.

    Singleton — use get_instance() rather than constructing directly.
    """

    _instance: Optional["LocalEmbedder"] = None
    _model = None

    def __init__(self):
        """Load the sentence-transformer model once."""
        self._logger = get_logger("embedder")
        self._CONFIG = Config.get_instance()
        self._load_model()

    def name(self) -> str:
        """Return the embedding function identifier for ChromaDB compatibility."""
        return self._CONFIG.EMBEDDING_MODEL  # ← THIS IS THE FIX

    def _load_model(self) -> None:
        """Load all-MiniLM-L6-v2 from sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._CONFIG.EMBEDDING_MODEL)
            self._logger.info(
                "Embedding model loaded.", model=self._CONFIG.EMBEDDING_MODEL, dim=self._CONFIG.EMBEDDING_DIM
            )
        except Exception as exc:
            self._logger.error("Failed to load embedding model", error=str(exc))
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts — called by ChromaDB on store and query.

        Parameters
        ----------
        input : List[str]
            Texts to embed.

        Returns
        -------
        List[List[float]]
            List of 384-dimensional float vectors.
        """
        if not input:
            return []
        try:
            vectors: np.ndarray = self._model.encode(
                input, show_progress_bar=False, convert_to_numpy=True
            )
            return vectors.tolist()
        except Exception as exc:
            self._logger.error("Embedding failed", error=str(exc), batch_size=len(input))
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string for retrieval.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        List[float]
            384-dimensional float vector.
        """
        return self([text])[0]

    @classmethod
    def get_instance(cls) -> "LocalEmbedder":
        """
        Return the singleton LocalEmbedder, loading the model if needed.

        Returns
        -------
        LocalEmbedder
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Load at import time so subsequent cells don't wait
EMBEDDER = LocalEmbedder.get_instance()


# ----------------------------------------------------------------------------
# Cell 7: Embedding Model
# Purpose: Provide a singleton sentence-transformer embedder for ChromaDB.
# Key Classes: LocalEmbedder
# Key Functions:
#   LocalEmbedder.__call__(input: List[str]) → List[List[float]]
#   LocalEmbedder.embed_query(text: str) → List[float]
#   LocalEmbedder.get_instance() → LocalEmbedder
# Key Constants/Config: CONFIG.EMBEDDING_MODEL, CONFIG.EMBEDDING_DIM
# Imports exported: LocalEmbedder, EMBEDDER
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger)
# Critical notes: EMBEDDER is the shared singleton — pass it to ChromaDB
#   collection constructors as the embedding_function parameter.
#   __call__ signature matches chromadb.api.types.EmbeddingFunction.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------

# =============================================================================
# CELL 8 — ChromaDB Store
# =============================================================================
"""
ChromaStore: single access point for all four ChromaDB collections.
Wraps every external call with the CB_CHROMADB circuit breaker.
Singleton pattern.
"""

from typing import Dict, List, Optional, Any
import chromadb
from logger import get_logger
from embedder import LocalEmbedder
from config import Config

class ChromaStore:
    """
    Manages the four ChromaDB collections used by the agent.

    Collections
    -----------
    child_chunks    : 400-token paragraphs (primary retrieval)
    parent_sections : 2500-token full sections (context expansion)
    financial_facts : atomic numeric facts
    mgmt_statements : chairman letter / MD&A / strategy content

    Singleton — use get_instance().
    """

    _instance: Optional["ChromaStore"] = None

    def __init__(self):
        """Initialise ChromaDB client and ensure all collections exist."""
        self._CONFIG = Config.get_instance()
        self._logger = get_logger("chroma_store")
        self._client = chromadb.PersistentClient(path=self._CONFIG.CHROMA_PATH)
        self._embedder = LocalEmbedder.get_instance()
        self._collections: Dict[str, chromadb.Collection] = {}
        self._init_collections()


    def _init_collections(self) -> None:
        """Create or retrieve all four collections."""
        names = [self._CONFIG.COL_CHILD, self._CONFIG.COL_PARENT, self._CONFIG.COL_FACTS, self._CONFIG.COL_MGMT]
        for name in names:
            col = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedder,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[name] = col
            self._logger.info("Collection ready.", collection=name, count=col.count())

    def query_collection(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query a collection with circuit-breaker protection.

        Parameters
        ----------
        collection_name : str
            One of the four collection name constants from Config.
        query_texts : List[str]
            Query strings to embed and search.
        n_results : int
            Maximum results to return per query.
        where : dict, optional
            ChromaDB metadata filter.

        Returns
        -------
        dict
            ChromaDB query result dict with keys: ids, documents, metadatas, distances.
        """
        col = self._collections.get(collection_name)
        if col is None:
            self._logger.error("Unknown collection", name=collection_name)
            raise ValueError(f"Unknown collection: {collection_name}")

        def _do_query():
            kwargs: Dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": min(n_results, max(col.count(), 1)),
            }
            if where:
                kwargs["where"] = where
            return col.query(**kwargs)

        try:
            return CB_CHROMADB.call(_do_query)
        except CircuitBreakerOpenError as exc:
            self._logger.error("ChromaDB circuit open", error=str(exc))
            raise
        except Exception as exc:
            self._logger.error("ChromaDB query failed", collection=collection_name, error=str(exc))
            raise

    def upsert_batch(self, collection_name: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        """
        Idempotent batch upsert into a collection.

        Parameters
        ----------
        collection_name : str
            Target collection name.
        ids : List[str]
            Chunk IDs (format: {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}).
        documents : List[str]
            Text content of each chunk.
        metadatas : List[Dict]
            Metadata dicts (one per chunk).
        """
        if not ids:
            return
        col = self._collections.get(collection_name)
        if col is None:
            raise ValueError(f"Unknown collection: {collection_name}")

        # Process in batches of CONFIG.DB_BATCH_SIZE
        for start in range(0, len(ids), self._CONFIG.DB_BATCH_SIZE):
            batch_ids = ids[start: start + self._CONFIG.DB_BATCH_SIZE]
            batch_docs = documents[start: start + self._CONFIG.DB_BATCH_SIZE]
            batch_meta = metadatas[start: start + self._CONFIG.DB_BATCH_SIZE]
            try:
                CB_CHROMADB.call(col.upsert, ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            except Exception as exc:
                self._logger.error(
                    "Upsert batch failed",
                    collection=collection_name,
                    batch_start=start,
                    error=str(exc),
                )
                raise
        self._logger.info("Upsert complete.", collection=collection_name, total=len(ids))

    def get_by_id(self, collection_name: str, chunk_id: str) -> Optional[Dict]:
        """
        Fetch a single chunk by its ID.

        Parameters
        ----------
        collection_name : str
            Collection to search.
        chunk_id : str
            Exact chunk ID.

        Returns
        -------
        dict or None
            {'id', 'document', 'metadata'} or None if not found.
        """
        col = self._collections.get(collection_name)
        if col is None:
            return None
        try:
            result = CB_CHROMADB.call(col.get, ids=[chunk_id])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as exc:
            self._logger.error("get_by_id failed", id=chunk_id, error=str(exc))
        return None

    def status(self) -> str:
        """
        Return a human-readable summary of chunk counts per collection.

        Returns
        -------
        str
        """
        lines = ["=== ChromaDB Status ==="]
        for name, col in self._collections.items():
            lines.append(f"  {name}: {col.count()} chunks")
        return "\n".join(lines)

    @classmethod
    def get_instance(cls) -> "ChromaStore":
        """Return the singleton ChromaStore, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


CHROMA_STORE = ChromaStore.get_instance()
print(CHROMA_STORE.status())

# ----------------------------------------------------------------------------
# Cell 8: ChromaDB Store
# Purpose: Manage all ChromaDB collections with circuit-breaker protection.
# Key Classes: ChromaStore
# Key Functions:
#   ChromaStore.query_collection(name, query_texts, n_results, where) → dict
#   ChromaStore.upsert_batch(name, ids, documents, metadatas) → None
#   ChromaStore.get_by_id(name, chunk_id) → dict | None
#   ChromaStore.status() → str
#   ChromaStore.get_instance() → ChromaStore
# Key Constants/Config: CONFIG.COL_CHILD/PARENT/FACTS/MGMT, CONFIG.CHROMA_PATH,
#   CONFIG.DB_BATCH_SIZE, CB_CHROMADB
# Imports exported: ChromaStore, CHROMA_STORE
# Depends on: Cell 3 (CONFIG), Cell 4 (get_logger), Cell 5 (CB_CHROMADB,
#   CircuitBreakerOpenError), Cell 7 (EMBEDDER)
# Critical notes: All four collections are created at init time.
#   query_collection caps n_results to col.count() to avoid ChromaDB error
#   when the collection has fewer items than n_results.
# Context Update: None
# Status: Complete
# ----------------------------------------------------------------------------
