**About Project**
AI-Agent-powered web application that enables investors to analyse stocks across multiple dimensions - financial health, corporate governance, market sentiment, future ambitions, and external macro factors - to identify quality investment opportunities aligned with user-defined preferences and risk appetite.			

**Tech Stack for Phase 1 (Google Colab)**
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

**Robustness Check**
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

**Naming Conventions**
Files:     snake_case.py
Classes:   PascalCase
Functions: snake_case
Constants: UPPER_SNAKE_CASE
Variables: snake_case
Chunk IDs: {SCRIP}_{FY}_{SECTION}_{PAGE}_{INDEX}        
Log files: {date}_{component}.log
Database tables: plural snake_case

**Coding Conventions**
Every function must have a docstring
If you are writing "and" in a function name: process_and_store_pdf()→ Split into process_pdf() and store_pdf()
No hardcoded values — use config.py

**Database Design Rules**
Every table must have id, created_at, updated_at
Never delete records — use is_deleted flag instead
All foreign keys must have indexes

**Error Handling Rules**
Never use bare except — always catch specific errors
All errors must be logged before raising
API endpoints must return proper HTTP status codes

**Phase 1 built**
Cell 1:  Install all dependencies
Cell 2:  Mount Drive + create folder structure under /brain/
Cell 3:  Config — all settings in one place
Cell 4:  Logger — structured JSON logs, not print statements
Cell 5:  Error handling — retry, circuit breaker, classification
Cell 6:  Input validator — every input checked before processing
Cell 7:  Embedding model — local, no quota
Cell 8:  ChromaDB — 4 collections with proper setup
Cell 9:  PDF classifier — detect native vs scanned
Cell 10: Section detector — 28 patterns for annual report sections
Cell 11: Table extractor — structure-preserving
Cell 12: Hierarchical chunker — parent + child + facts
Cell 13: Deduplication + storage manager
Cell 14: BM25 keyword index
Cell 15: Hybrid retriever — semantic + BM25 + RRF
Cell 16: Query understanding — classify, decompose, expand
Cell 17: Working memory
Cell 18: Gap analyser
Cell 19: Synthesis engine + verification
Cell 20: REACT agent — full orchestration
Cell 21: Process your PDFs
Cell 22: Query the agent
Cell 23: Manual test suite

**Instruction For Claude**
After writing EVERY cell, output a completion summary that can be used by any other agent to completely understand code. Format

Cell [N]: [Cell Name]
Purpose: [What this cell does in one line]
Key Classes: [ClassName]
Key Functions: [function_name(param1, param2) → return_type] 
Key Constants/Config: [CONSTANT_NAME, config_key]  
Imports exported: [what other cells will import FROM this cell] 
Depends on: [which previous cells this cell imports from] 
Critical notes: [anything next cell MUST know — patterns used, gotchas, design decisions]
Context Update : [Anythink that is useful for context update in general]
Status: Complete

