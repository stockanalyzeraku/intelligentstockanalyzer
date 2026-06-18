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
    MISTRAL_MODEL_OCR = "mistral-ocr-latest"
    GEMINI_API_KEY: Optional[str] = ""
    MISTRAL_API_KEY : Optional[str] = ""
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_OUTPUT_TOKENS: int = 4096
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str =  "all-MiniLM-L6-v2"
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
CONFIG = Config.get_instance()
#print(CONFIG)
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
