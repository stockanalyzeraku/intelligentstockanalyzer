Tech Stack for Phase 1 (Google Colab)
LLM :	Gemini 2.5 Flash
Embeddings : all-MiniLM-L6-v2
Vector DB	: ChromaDB (local)
Primary DB	: SQLite
Cache	: In-memory dict
Task Queue	: None (sync)
Auth	: None
File Storage :	Google Drive
Backend : API	Notebook calls
Frontend :	None
Hosting	: Google Colab
Logging	: Print statements
LLM Observ. :	None
Testing :	Manual
CI/CD	: None
Version Ctrl	: GitHub

Include Robustness in each file. Check for:
Structured logging (not print statements)
Error classification — transient vs permanent
Retry with exponential backoff
Circuit breaker on all external calls
Input validation at every boundary
Thread safety where shared state exists
Deduplication and idempotency
Memory bounds on all collections
Timeouts on all external calls
No silent failures — every exception logged
