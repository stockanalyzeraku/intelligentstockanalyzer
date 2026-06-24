"""
Central configuration for the financials package.

Keep all "magic" constants here so scraper.py, db.py, ingest.py etc. stay
readable and so behaviour can be tuned in one place.
"""
import sys
import os

from config import CONFIG
import os

# ---------------------------------------------------------------------------
# Screener.in
# ---------------------------------------------------------------------------

SCREENER_BASE_URL = "https://www.screener.in"

# Screener accepts the NSE trading symbol directly in this URL pattern.
# e.g. https://www.screener.in/company/RELIANCE/consolidated/
SCREENER_CONSOLIDATED_URL_TMPL = SCREENER_BASE_URL + "/company/{symbol}/consolidated/"


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT_SECONDS = 20
ANNUAL_YEARS_TO_KEEP = 5

STATEMENT_SECTIONS = {
    "profit_loss": "Profit & Loss",
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flows",
}


DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CONFIG.FINANCIALS_DB_PATH, "financials.db")

