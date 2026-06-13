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


def retry(self,
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
