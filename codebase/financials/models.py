"""
Plain data containers passed between the scraper and the storage layer.

Keeping these as simple dataclasses (rather than passing raw dicts around)
makes the hand-off between scraper.py -> ingest.py -> db.py explicit and
type-checkable.
"""
import sys
import os

from dataclasses import dataclass, field


@dataclass
class StatementRow:
    """One line item (row) of a financial statement for all scraped years.

    Example: line_item="Sales", values={"Mar 2024": 899041.0, "Mar 2025": 962820.0}
    """
    line_item: str
    values: dict  # {period_label: float_or_None}


@dataclass
class StatementTable:
    """A full statement (P&L / Balance Sheet / Cash Flow) for one company."""
    statement_key: str          # e.g. "profit_loss"
    statement_label: str        # e.g. "Profit & Loss"
    periods: list               # ordered list of period labels e.g. ["Mar 2022", ..., "Mar 2026"]
    rows: list = field(default_factory=list)  # list[StatementRow]


@dataclass
class CompanyInfo:
    """Identity info scraped/derived for the company."""
    input_symbol: str           # what the caller passed in
    screener_symbol: str        # symbol used in the screener.in URL
    company_name: str
    nse_symbol: str = None
    bse_code: str = None
    source_url: str = None


@dataclass
class ScrapedCompanyData:
    """Everything scraped for one company in one run."""
    company: CompanyInfo
    statements: list            # list[StatementTable]
