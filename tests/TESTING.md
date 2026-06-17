# Testing Plan for Intelligent Stock Analyzer

This document is written in table format so it can be copied into Excel/Sheets.

## Test Folder Structure

| Folder | Purpose | Notes |
|---|---|---|
| `tests/unit/` | Fast tests for one class/module at a time | No real Chroma, no LLM, no large PDFs |
| `tests/integration/` | Tests multiple modules working together | Prefer fake Chroma first, real Chroma later |
| `tests/helpers/` | Reusable fake stores and test helpers | Keeps tests clean and avoids duplication |
| `tests/fixtures/` | Small static test data | Sample JSON records, cleaned pages, embedding-ready payloads |
| `tests/golden/` | Golden financial Q&A regression tests | Add after the core unit/integration tests are stable |

## Current Automated Tests

| Test File | Level | Module(s) Covered | Main Behavior Checked | External Dependencies |
|---|---|---|---|---|
| `tests/unit/test_cachememory.py` | Unit | `CacheMemory` | Cache key stability, cache insert/read, hit count, expiry | Temporary SQLite DB only |
| `tests/unit/test_workingmemory.py` | Unit | `WorkingMemory` | PDF registration, duplicate upsert, artifacts, Chroma store metadata, list/get helpers | Temporary SQLite DB and temp PDF bytes |
| `tests/unit/test_querycheckpointer.py` | Unit | `QueryCheckpointer` | Rejects basic metric queries and accepts specific company/year queries | None |
| `tests/unit/test_queryplanner.py` | Unit | `FinancialQueryPlanner` | Financial synonym expansion and filter creation | None |
| `tests/unit/test_contextbuilder.py` | Unit | `FinancialContextBuilder` | Builds context text and citations from fake retrieval records | Fake record only |
| `tests/unit/test_citationdebugger.py` | Unit | `CitationDebugWriter` | Writes readable JSON with tools and citations | Temporary directory only |
| `tests/integration/test_financialpipelinerunner_fake_store.py` | Integration | `FinancialPipelineRunner`, `CacheMemory`, fake Chroma store | Checkpointer bypass, cache miss, retrieval, cache hit, debug JSON | Temporary SQLite DB and fake Chroma store |

## Recommended Commands

| Command | Purpose | When To Run |
|---|---|---|
| `python -m py_compile codebase/agentmemory/*.py codebase/agentrunpipeline/*.py codebase/vectordb/chromastore.py` | Syntax check for main app modules | Before every commit |
| `python -m pytest tests/unit -q` | Run fast unit tests | During development |
| `python -m pytest tests/integration -q` | Run module integration tests | Before committing pipeline changes |
| `python -m pytest tests -q` | Run all tests | Before opening PRs |

## Manual Future Test Ideas

| Area | Test Idea | Expected Result |
|---|---|---|
| Real Chroma retrieval | Create a tiny temporary Chroma DB with parent/child records | Correct parent page is returned for revenue/risk queries |
| Golden questions | Store 20-50 known financial questions with expected pages/terms | RAG answer contains expected terms and citations |
| Cache invalidation | Reprocess same company/year and invalidate old cache | Old cached answer is not reused |
| Debug tracing | Inspect debug JSON for fresh and cached answers | Fresh run shows cache lookup + retrieval; cache hit shows cache lookup only |
| Answer quality | Compare generated answer against source citation | Answer is supported by cited context |

## First Tests to Add Next

| Priority | Test | Why |
|---|---|---|
| 1 | Real small Chroma DB integration test | Validates retrieval behavior beyond fake records |
| 2 | Golden question fixture for one company/year | Prevents future financial-answer regressions |
| 3 | Cache invalidation by company/year | Needed when reports are reprocessed |
| 4 | CLI/script smoke tests | Useful for manual checks without remembering pytest commands |
| 5 | End-to-end sample PDF processing test | Validates PDF-to-Chroma-to-RAG flow |

## Testing Rules

| Rule | Reason |
|---|---|
| Use temporary DB paths in tests | Avoids modifying `database/brain.db` |
| Use fake Chroma for fast runner tests | Avoids model/vector dependencies |
| Keep real Chroma tests separate | They are slower and more environment-sensitive |
| Do not commit generated debug JSON | Debug output should be temporary |
| Do not commit generated cache DB files | Tests should be reproducible from source |
| Prefer small fixtures | Keeps tests fast and readable |
