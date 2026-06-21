"""
Central configuration for the financials package.

Keep all "magic" constants here so scraper.py, db.py, ingest.py etc. stay
readable and so behaviour can be tuned in one place.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os

# ---------------------------------------------------------------------------
# Screener.in
# ---------------------------------------------------------------------------

SCREENER_BASE_URL = "https://www.screener.in"

# Screener accepts the NSE trading symbol directly in this URL pattern.
# e.g. https://www.screener.in/company/RELIANCE/consolidated/
SCREENER_CONSOLIDATED_URL_TMPL = SCREENER_BASE_URL + "/company/{symbol}/consolidated/"

# Some companies (BSE-only listings, or companies screener tracks by BSE code)
# don't have an NSE symbol and screener falls back to the numeric BSE code in
# the URL, e.g. /company/500325/ (standalone-only, no /consolidated/ page
# exists for those). We do NOT auto-resolve this (per product decision) -
# the caller must supply the exact screener.in symbol/slug.

REQUEST_HEADERS = {
    # A realistic desktop UA. Screener doesn't require this, but some hosts
    # block the default python-requests UA.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT_SECONDS = 20

# ---------------------------------------------------------------------------
# Scope of data we scrape
# ---------------------------------------------------------------------------

# Only annual columns are kept (column headers look like "Mar 2024").
# Screener shows up to ~12 annual columns; we only persist the most recent N.
ANNUAL_YEARS_TO_KEEP = 5

# The statement sections we scrape, keyed by an internal "statement_key"
# (used as part of the table name) -> the heading text screener.in uses for
# that section on the consolidated company page.
STATEMENT_SECTIONS = {
    "profit_loss": "Profit & Loss",
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flows",
}

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "financials.db")
