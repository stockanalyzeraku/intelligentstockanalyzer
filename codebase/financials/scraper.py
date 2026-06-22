"""
Scraper for screener.in consolidated company pages.

Responsibilities:
  1. Fetch the consolidated company page for a given NSE/BSE symbol.
  2. Parse out company identity (name, NSE/BSE codes).
  3. Parse the Profit & Loss, Balance Sheet and Cash Flow HTML tables.
  4. Restrict each table to the most recent N annual columns
     (annual columns are headed like "Mar 2024" - screener marks quarterly
     columns differently, e.g. "Mar 2024" under "Quarterly Results" section,
     so we scope our search to the right section of the page first).

This module deliberately does NOT touch SQLite - it only returns clean
ScrapedCompanyData. Persistence lives in db.py / ingest.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re

import requests
from bs4 import BeautifulSoup

from codebase.financials import config
from codebase.financials.models import CompanyInfo, ScrapedCompanyData, StatementRow, StatementTable


class ScrapeError(Exception):
    """Raised when the page can't be fetched or doesn't look like a company page."""


def _clean_number(text):
    """Convert a screener.in numeric cell to a float, or None if blank/NA.

    Handles: commas ("1,05,780"), parentheses for negatives ("(1,234)"),
    percent signs ("18%"), trailing/leading whitespace, and the unicode
    minus sign sometimes used instead of a hyphen.
    """
    if text is None:
        return None
    t = text.strip().replace("\u2212", "-")
    if t in ("", "-", "—"):
        return None
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative = True
        t = t[1:-1]
    t = t.replace(",", "").replace("%", "").strip()
    if t in ("", "-"):
        return None
    try:
        value = float(t)
    except ValueError:
        return None
    return -value if negative else value


def _clean_label(text):
    """Strip the trailing '+' expand marker and collapse whitespace in a row label."""
    if text is None:
        return ""
    t = " ".join(text.split())
    t = re.sub(r"\s*\+\s*$", "", t).strip()
    return t


def fetch_page(symbol):
    """Fetch the raw HTML of the consolidated company page for `symbol`.

    Raises ScrapeError on network failure or a non-200 / non-company response.
    """
    url = config.SCREENER_CONSOLIDATED_URL_TMPL.format(symbol=symbol.strip().upper())
    try:
        resp = requests.get(
            url,
            headers=config.REQUEST_HEADERS,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ScrapeError(f"Network error fetching {url}: {exc}") from exc

    if resp.status_code == 404:
        raise ScrapeError(
            f"Symbol '{symbol}' not found on screener.in (404 at {url}). "
            f"Confirm the exact NSE trading symbol screener.in uses."
        )
    if resp.status_code != 200:
        raise ScrapeError(f"Unexpected status {resp.status_code} fetching {url}")

    return resp.text, url


def parse_company_info(html, input_symbol, source_url):
    """Extract company name, NSE symbol and BSE code from the page header."""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    company_name = h1.get_text(strip=True) if h1 else input_symbol

    nse_symbol = None
    bse_code = None

    # Screener renders small links like "BSE: 500325" and "NSE: RELIANCE"
    # near the top of the page.
    for a in soup.find_all("a"):
        txt = a.get_text(strip=True)
        m_bse = re.match(r"^BSE:\s*(\S+)$", txt)
        m_nse = re.match(r"^NSE:\s*(\S+)$", txt)
        if m_bse:
            bse_code = m_bse.group(1)
        elif m_nse:
            nse_symbol = m_nse.group(1)

    screener_symbol = input_symbol.strip().upper()

    if not company_name or company_name == input_symbol:
        raise ScrapeError(
            f"Could not confidently parse a company page for '{input_symbol}'. "
            f"The page may not exist or screener.in's layout has changed."
        )

    return CompanyInfo(
        input_symbol=input_symbol,
        screener_symbol=screener_symbol,
        company_name=company_name,
        nse_symbol=nse_symbol,
        bse_code=bse_code,
        source_url=source_url,
    )


def _find_section_table(soup, heading_text):
    """Find the <table> belonging to the section whose <h2> matches heading_text.

    Screener structures each section as:
        <section id="profit-loss"> ... <h2>Profit &amp; Loss</h2> ... <table>...</table> ...</section>
    but section ids/markup can shift, so we walk forward from the matching
    heading to the next <table> in document order, which is robust to minor
    structural changes.
    """
    heading = None
    for tag in soup.find_all(["h2", "h1"]):
        if tag.get_text(strip=True).lower().startswith(heading_text.lower()):
            heading = tag
            break

    if heading is None:
        raise ScrapeError(f"Could not locate section heading '{heading_text}' on page.")

    table = heading.find_next("table")
    if table is None:
        raise ScrapeError(f"Found heading '{heading_text}' but no following <table>.")

    return table


def _parse_statement_table(table_tag, statement_key, statement_label, years_to_keep):
    """Turn a screener <table> into a StatementTable, keeping only the most
    recent `years_to_keep` annual columns.

    Screener's annual tables have a header row like:
        ["", "Mar 2015", "Mar 2016", ..., "Mar 2026"]
    and each body row like:
        ["Sales +", "374,372", "272,583", ..., "1,055,780"]
    """
    thead = table_tag.find("thead")
    header_cells = thead.find_all("th") if thead else table_tag.find("tr").find_all("th")
    all_periods = [th.get_text(strip=True) for th in header_cells][1:]  # skip blank corner cell

    if not all_periods:
        raise ScrapeError(f"No period columns found for section '{statement_label}'.")

    # Keep only the most recent N columns (rightmost = most recent on screener.in).
    keep_periods = all_periods[-years_to_keep:]
    keep_count = len(keep_periods)

    tbody = table_tag.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table_tag.find_all("tr")[1:]

    rows = []
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        label = _clean_label(cells[0].get_text(strip=True))
        if not label:
            continue

        raw_values = [c.get_text(strip=True) for c in cells[1:]]
        # Align to the same trailing slice as the header (defensive: some
        # rows, e.g. with footnote markers, can have fewer cells).
        raw_values = raw_values[-keep_count:] if len(raw_values) >= keep_count else raw_values
        values = {}
        for i, period in enumerate(keep_periods):
            values[period] = _clean_number(raw_values[i]) if i < len(raw_values) else None

        rows.append(StatementRow(line_item=label, values=values))

    return StatementTable(
        statement_key=statement_key,
        statement_label=statement_label,
        periods=keep_periods,
        rows=rows,
    )


def scrape_company(symbol, years_to_keep=None):
    """Full scrape entry point: fetch + parse company info and all statements.

    Parameters
    ----------
    symbol : str
        The NSE trading symbol as used by screener.in
        (e.g. "RELIANCE", "TCS", "INFY").
    years_to_keep : int, optional
        Number of most-recent annual columns to retain. Defaults to
        config.ANNUAL_YEARS_TO_KEEP.

    Returns
    -------
    ScrapedCompanyData
    """
    years_to_keep = years_to_keep or config.ANNUAL_YEARS_TO_KEEP

    html, source_url = fetch_page(symbol)
    soup = BeautifulSoup(html, "html.parser")

    company = parse_company_info(html, symbol, source_url)

    statements = []
    for statement_key, heading_text in config.STATEMENT_SECTIONS.items():
        table_tag = _find_section_table(soup, heading_text)
        statement = _parse_statement_table(
            table_tag, statement_key, heading_text, years_to_keep
        )
        statements.append(statement)

    return ScrapedCompanyData(company=company, statements=statements)
