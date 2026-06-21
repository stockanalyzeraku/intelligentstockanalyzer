#!/usr/bin/env python3
"""
Standalone debug/explore script for the financials package.

This file is intentionally separate from the library code (db.py,
discovery.py, ingest.py) so you can run it directly to poke at the database
and schema without writing any code of your own - useful while developing,
debugging scraper changes, or just inspecting what's in financials.db.

USAGE (from inside codebase/financials/, or anywhere with this package on
PYTHONPATH):

    # Show every logical table + description (the discovery catalogue)
    python3 debug_explore.py tables

    # Show columns for one table
    python3 debug_explore.py describe statement_profit_loss

    # List every company currently stored
    python3 debug_explore.py companies

    # Scrape + store a company (calls ingest_company under the hood)
    python3 debug_explore.py ingest RELIANCE

    # Dump one company's full P&L / Balance Sheet / Cash Flow (pivoted by year)
    python3 debug_explore.py dump RELIANCE

    # Dump just one statement
    python3 debug_explore.py dump RELIANCE --table statement_cash_flow

    # List which line items exist for a company/table
    python3 debug_explore.py line-items RELIANCE statement_balance_sheet

    # Run an arbitrary read-only SQL query
    python3 debug_explore.py sql "SELECT * FROM companies"

If a company hasn't been ingested yet, `dump` / `line-items` will tell you
to run `ingest` first.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import sys
from pathlib import Path

# Allow running this file directly (``python3 debug_explore.py ...``) by
# making sure the parent of codebase/ is importable, regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # .../codebase/financials -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codebase.financials import db, discovery  # noqa: E402
from codebase.financials.ingest import ingest_company  # noqa: E402
from codebase.financials.scraper import ScrapeError  # noqa: E402


def _print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_tables(args):
    db.init_schema()
    _print_json(discovery.list_tables())


def cmd_describe(args):
    db.init_schema()
    try:
        _print_json(discovery.describe_table(args.table_name))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_companies(args):
    db.init_schema()
    _print_json(discovery.list_companies())


def cmd_ingest(args):
    db.init_schema()
    try:
        result = ingest_company(args.symbol, years_to_keep=args.years)
        _print_json(result)
    except ScrapeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _resolve_company_or_die(symbol):
    company = discovery.find_company(symbol)
    if not company:
        print(
            f"No stored data for '{symbol}' yet. Run:\n"
            f"    python3 debug_explore.py ingest {symbol}\n"
            f"first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return company


def cmd_dump(args):
    db.init_schema()
    company = _resolve_company_or_die(args.symbol)
    company_id = company["company_id"]

    try:
        if args.table:
            data = discovery.get_statement(args.table, company_id, pivot=not args.long)
            _print_json({"company": company, args.table: data})
        else:
            data = discovery.get_all_statements_for_company(company_id, pivot=not args.long)
            _print_json({"company": company, **data})
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_line_items(args):
    db.init_schema()
    company = None
    company_id = None
    if args.symbol:
        company = _resolve_company_or_die(args.symbol)
        company_id = company["company_id"]
    try:
        items = discovery.list_line_items(args.table_name, company_id=company_id)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    _print_json({"table": args.table_name, "company": company, "line_items": items})


def cmd_sql(args):
    db.init_schema()
    try:
        rows = discovery.run_raw_query(args.query)
        _print_json(rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tables", help="List every logical table + description")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("describe", help="Show columns for one table")
    p.add_argument("table_name")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("companies", help="List every company stored in the DB")
    p.set_defaults(func=cmd_companies)

    p = sub.add_parser("ingest", help="Scrape + store a company from screener.in")
    p.add_argument("symbol", help="NSE trading symbol as used by screener.in, e.g. RELIANCE")
    p.add_argument("--years", type=int, default=None, help="How many recent annual years to keep (default: 5)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("dump", help="Dump stored statement data for a company")
    p.add_argument("symbol")
    p.add_argument("--table", default=None, help="Limit to one statement table, e.g. statement_cash_flow")
    p.add_argument("--long", action="store_true", help="Return tidy long rows instead of pivoted-by-year")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("line-items", help="List line items available for a table (optionally scoped to a company)")
    p.add_argument("symbol", nargs="?", default=None, help="Optional company symbol to scope to")
    p.add_argument("table_name")
    p.set_defaults(func=cmd_line_items)

    p = sub.add_parser("sql", help="Run an arbitrary read-only SQL SELECT query")
    p.add_argument("query")
    p.set_defaults(func=cmd_sql)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
