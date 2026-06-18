# =============================================================================
# CELL 5 — Error Handler
# =============================================================================
"""
Exception hierarchy, error classification, retry-with-backoff, timeout helpers,
and circuit breakers for external/fragile operations.
"""

from __future__ import annotations

import concurrent.futures
import functools
import random
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Type

from config import CONFIG, Config
from logger import get_logger


class ErrorType(Enum):
    """Classification of errors by recoverability."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class AppError(Exception):
    """Base application error carrying structured diagnostic metadata."""

    error_code = "APP_ERROR"
    retryable = False
    recoverable = False

    def __init__(self, message: str, *, safe_message: str | None = None, details: dict[str, Any] | None = None, cause: BaseException | None = None):
        super().__init__(message)
        self.safe_message = safe_message or message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "safe_message": self.safe_message,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "details": self.details,
        }


class ValidationError(AppError, ValueError):
    error_code = "VALIDATION_ERROR"


class PromptInjectionError(ValidationError):
    error_code = "PROMPT_INJECTION_DETECTED"


class PathSafetyError(ValidationError):
    error_code = "PATH_SAFETY_ERROR"


class FileTooLargeError(ValidationError):
    error_code = "FILE_TOO_LARGE"


class InvalidSchemaError(ValidationError):
    error_code = "INVALID_SCHEMA"


class UnsafeQueryFilterError(ValidationError):
    error_code = "UNSAFE_QUERY_FILTER"


class ProcessingError(AppError):
    error_code = "PROCESSING_ERROR"
    recoverable = True


class RetrievalError(AppError):
    error_code = "RETRIEVAL_ERROR"
    retryable = True
    recoverable = True


class LLMError(AppError):
    error_code = "LLM_ERROR"
    retryable = True
    recoverable = True


class LLMTimeoutError(LLMError, TimeoutError):
    error_code = "LLM_TIMEOUT"


class DataIntegrityError(AppError):
    error_code = "DATA_INTEGRITY_ERROR"


class DependencyError(AppError):
    error_code = "DEPENDENCY_ERROR"
    retryable = True
    recoverable = True


class CircuitBreakerOpenError(DependencyError):
    """Raised when a call is attempted while the circuit breaker is OPEN."""

    error_code = "CIRCUIT_BREAKER_OPEN"


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker implementing CLOSED → OPEN → HALF_OPEN."""

    def __init__(self, name: str, failure_threshold: int, recovery_timeout: int):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        self._logger = get_logger("circuit_breaker").bind(breaker=name)

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if (
                self._state == CircuitBreakerState.OPEN
                and self._last_failure_time is not None
                and time.monotonic() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
                self._logger.info("Circuit breaker transitioned to HALF_OPEN", event="circuit_half_open")
            return self._state

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        current = self.state
        if current == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self._name}' is OPEN — call rejected.")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            if classify_error(exc) == ErrorType.TRANSIENT:
                self._on_failure(exc)
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._logger.info("Circuit breaker recovered to CLOSED", event="circuit_closed")

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._logger.error(
                "Circuit breaker recorded failure",
                event="circuit_failure",
                failure_count=self._failure_count,
                exception_type=exc.__class__.__name__,
                error=str(exc),
            )
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._logger.error("Circuit breaker OPENED", event="circuit_opened", failure_count=self._failure_count)

    def status(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }


def classify_error(exc: Exception) -> ErrorType:
    """Classify an exception as TRANSIENT or PERMANENT."""
    if isinstance(exc, AppError):
        return ErrorType.TRANSIENT if exc.retryable else ErrorType.PERMANENT
    transient_types = (TimeoutError, ConnectionError, OSError, concurrent.futures.TimeoutError)
    transient_messages = ("rate limit", "quota", "503", "502", "500", "429", "timeout", "temporarily", "unavailable", "connection reset")
    if isinstance(exc, transient_types):
        return ErrorType.TRANSIENT
    msg = str(exc).lower()
    return ErrorType.TRANSIENT if any(kw in msg for kw in transient_messages) else ErrorType.PERMANENT


def retry(
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: retry transient errors using exponential backoff with full jitter."""
    _logger = get_logger("retry")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = max_attempts or CONFIG.RETRY_MAX_ATTEMPTS
            b_delay = base_delay or CONFIG.RETRY_BASE_DELAY
            m_delay = max_delay or CONFIG.RETRY_MAX_DELAY
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if classify_error(exc) == ErrorType.PERMANENT:
                        _logger.error("Permanent error — not retrying", event="retry_permanent_error", func=func.__name__, error=str(exc))
                        raise
                    if attempt == attempts:
                        break
                    upper = min(m_delay, b_delay * (2 ** (attempt - 1)))
                    delay = random.uniform(0, upper)
                    _logger.warning(
                        "Transient error — retrying",
                        event="retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=attempts,
                        delay_s=round(delay, 2),
                        error=str(exc),
                    )
                    time.sleep(delay)
            _logger.error("All retry attempts exhausted", event="retry_exhausted", func=func.__name__, attempts=attempts, error=str(last_exc))
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def with_timeout(seconds: float, timeout_error: Type[Exception] = TimeoutError) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run a callable in a worker thread and raise timeout_error on deadline."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=seconds)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise timeout_error(f"{func.__name__} timed out after {seconds}s") from exc
            finally:
                if future.done():
                    executor.shutdown(wait=False, cancel_futures=True)
        return wrapper
    return decorator


def run_guarded(operation: str, func: Callable[..., Any], *, breaker: CircuitBreaker | None = None, timeout_s: float | None = None, retry_attempts: int | None = None, exception_type: Type[AppError] = AppError) -> Any:
    """Execute func with optional timeout, retry, circuit breaker, and structured logging."""
    logger = get_logger("guardrails")

    def call() -> Any:
        target = func
        if timeout_s:
            target = with_timeout(timeout_s)(target)
        if retry_attempts and retry_attempts > 1:
            target = retry(max_attempts=retry_attempts)(target)
        if breaker:
            return breaker.call(target)
        return target()

    start = time.perf_counter()
    try:
        result = call()
        logger.info(operation, event=f"{operation}_completed", duration_ms=round((time.perf_counter() - start) * 1000, 2))
        return result
    except AppError:
        raise
    except Exception as exc:
        logger.exception(operation, exc, event=f"{operation}_failed", duration_ms=round((time.perf_counter() - start) * 1000, 2))
        raise exception_type(f"{operation} failed: {exc}", cause=exc) from exc


# Pre-instantiated circuit breakers for shared services
CB_GEMINI = CircuitBreaker("gemini", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)
CB_MISTRAL = CircuitBreaker("mistral", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)
CB_CHROMADB = CircuitBreaker("chromadb", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)
CB_EMBEDDER = CircuitBreaker("embedder", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)
CB_FILESYSTEM = CircuitBreaker("filesystem", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)
CB_SQLITE = CircuitBreaker("sqlite", Config.CB_FAILURE_THRESHOLD, Config.CB_RECOVERY_TIMEOUT)

_eh_logger = get_logger("error_handler")
_eh_logger.info(
    "Error handler initialised.",
    event="error_handler_ready",
    breakers=[CB_GEMINI.status(), CB_MISTRAL.status(), CB_CHROMADB.status(), CB_EMBEDDER.status(), CB_FILESYSTEM.status(), CB_SQLITE.status()],
)
