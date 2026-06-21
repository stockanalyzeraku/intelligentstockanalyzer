"""
financials
==========

Scrape consolidated annual financial statements (Profit & Loss, Balance
Sheet, Cash Flow) for NSE/BSE-listed companies from screener.in and store
them in SQLite using a hybrid storage style: typed statement tables plus
metadata/discovery tables that let an agent learn the schema at runtime.

Quick start
-----------
    from codebase.financials.ingest import ingest_company
    from codebase.financials import discovery

    ingest_company("RELIANCE")                      # scrape + store
    company = discovery.find_company("RELIANCE")
    pl = discovery.get_statement(
        "statement_profit_loss", company["company_id"], pivot=True
    )

See debug_explore.py for a standalone CLI to inspect the database directly.
"""

from codebase.financials import config  # noqa: F401
