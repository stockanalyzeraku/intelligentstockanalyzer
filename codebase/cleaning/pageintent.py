"""
cleaning/pageintent.py
======================
Rule-based page-intent tagger for annual report pages.
"""


import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
 
import re
from dataclasses import dataclass, field
from logger import get_logger
from codebase.cleaning.cleanresult import CleanResult

logger = get_logger(__name__)


def _headings(page: CleanResult) -> list[str]:
    """Extract all markdown heading texts (# / ## / ###) from clean_text."""
    return re.findall(r'^#{1,3}\s+(.+)$', page.clean_text, re.MULTILINE)

def _text(page: CleanResult) -> str:
    return page.clean_text.lower()

def _has(page: CleanResult, *patterns) -> bool:
    t = _text(page)
    return any(re.search(p, t) for p in patterns)

def _heading_has(page: CleanResult, *patterns) -> bool:
    headings_lower = ' '.join(_headings(page)).lower()
    return any(re.search(p, headings_lower) for p in patterns)

def _has_currency_figures(page: CleanResult) -> bool:
    """True if page contains ₹ symbols or financial magnitude patterns."""
    t = page.clean_text
    return bool(re.search(r'₹|rs\.|million|crore|lakh|\d+\.\d{2}', t, re.IGNORECASE))

def _is_tabular(page: CleanResult) -> bool:
    return page.has_table

def _word_count(page: CleanResult) -> int:
    return page.word_count



INTENT_RULES: list[tuple[str, float, object]] = [

    # ── Cover / back-cover / branding ────────────────────────────────────
#    ("cover_page", 0.95, lambda p: (
#        p.get('is_short', False) and
#        _has(p, r'annual report')
#    )),
    ("back_cover", 0.90, lambda p: (
        _has(p, r'registered.*office|corporate office|cin:|tel:.*fax:')
    )),

    # ── Table of contents ─────────────────────────────────────────────────
    ("table_of_contents", 0.95, lambda p: (
        _heading_has(p, r"what.?s inside|contents|index") or
        (bool(re.search(r'\.\.\.\s*\d+', p.clean_text)) and
         _word_count(p) < 300)
    )),

    # ── Vision / Mission / Values ─────────────────────────────────────────
    ("vision_mission_values", 0.92, lambda p: (
        _heading_has(p, r'vision', r'mission', 
                     r'our values', r'core values')
    )),

    # ── MD's / Chairman's / Founder's message ────────────────────────────
    ("mds_message", 0.92, lambda p: (
        _heading_has(p, r"md.?s\s+(message|letter|address)",
                     r"managing director.?s\s+message",
                     r"rooted in values",
                     r"ready for tomorrow") or
        _has(p, r"managing director.*founder")
    )),
    ("chairmans_message", 0.92, lambda p: (
        _heading_has(p, r"chairman.?s\s+(message|letter|address|speech)",
                     r"from the chairman")
    )),

    # ── Company overview / About us ───────────────────────────────────────
    ("company_overview", 0.88, lambda p: (
        _heading_has(p,
                     r"a story crafted in trust",
                     r"company overview",
                     r"about us") or
        (_has(p, r"year legacy"))
    )),

    # ── Company timeline / History ────────────────────────────────────────
    ("company_timeline", 0.90, lambda p: (
        _heading_has(p, r"family legacy",
                     r"from a family legacy",
                     r"phases of growth",
                     r"our journey so far",
                     r"milestones") or
        (_is_tabular(p) and
         _has(p, r'first generation|second generation'))
    )),

    # ── Competitive strengths ─────────────────────────────────────────────
    ("competitive_strengths", 0.88, lambda p: (
        _heading_has(p, r"our\s+strengths",
                     r"kalyan edge",
                     r"competitive\s+(advantage|strength)") or
        (_has(p, r"leading brand.*large market",
              r"established brand.*trust.*transparency",
              r"pan-india presence"))
    )),

    # ── Retail footprint / Presence ───────────────────────────────────────
    ("retail_footprint_expansion", 0.88, lambda p: (
        _heading_has(p, r"our\s+presence",
                     r"retail\s+footprint",
                     r"taking the experience",
                     r"expansion") or
        (_has(p, r'foco|franchise.*owned.*company.*operated',
              r'showrooms',
              r'capital.light.*expansion'))
    )),

    # ── Product portfolio ─────────────────────────────────────────────────
    ("product_portfolio", 0.88, lambda p: (
        _heading_has(p, r"product\s+offerings",
                     r"collections\s+that\s+speak",
                     r"jewellery for every",
                     r"tradition.*reimagined.*design") or
        (_has(p, r'wedding jewellery|studded jewellery|aspirational jewellery',
              r'sub.brand|sub.collection'))
    )),

    # ── Performance highlights / Financial overview ───────────────────────
    ("performance_highlights", 0.90, lambda p: (
        _heading_has(p, r"performance\s+highlights",
                     r"financial\s+highlights",
                     r"growth with purpose") or
        (_is_tabular(p) and
         _has(p, r'fy25.*fy24.*fy23|fy21.*fy22.*fy23',
              r'total revenue.*million',
              r'pat.*pbt.*ebitda'))
    )),

    # ── Strategic initiatives / Strategic overview ────────────────────────
    ("strategic_overview", 0.88, lambda p: (
        _heading_has(p, r"strategic\s+(initiatives|overview|review)",
                     r"driving value",
                     r"strategic\s+priorities",
                     r"building deeper customer connections",
                     r"hyperlocal jeweller") or
        (_has(p, r'fy25\s+progress|fy26\s+plans|fy26\s+target',
              r'priorities.*expand.*capital.light'))
    )),

    # ── Marketing & brand promotions ──────────────────────────────────────
    ("marketing_promotions", 0.87, lambda p: (
        _heading_has(p, r"marketing\s+and\s+promotion",
                     r"brand\s+campaign",
                     r"brand\s+ambassador",
                     r"campaigns\s+that\s+celebrate",
                     r"digital meets tradition") or
                     (_has(p, r'brand ambassador|regional campaign',
                           r'marketing.*branding.*investment',
                           r'love is light')
                           ))),

    # ── Technology & digitalisation ───────────────────────────────────────
    ("technology_digitisation", 0.88, lambda p: (
        _heading_has(p, r"technology.driven",
                     r"using technology",
                     r"digitisation",
                     r"turning data into",
                     r"smart systems",
                     r"empowering our teams") or
        (_has(p, r'crm.*analytics|ai.powered|virtual try.on',
              r'erp.*barcode|demand.led stocking',
              r'omnichannel.*enablement'))
    )),

    # ── Management Discussion & Analysis ─────────────────────────────────
    ("mda", 0.93, lambda p: (
        _heading_has(p, r"management\s+discussion.*analysis",
                     r"md&a",
                     r"management.?s\s+discussion") or
        _has(p, r'management discussion.*analysis.*economic review',
             r'annexure.*management discussion')
    )),

    # ── Management team / Key personnel ──────────────────────────────────
    ("management_team", 0.88, lambda p: (
        _heading_has(p, r"management\s+team",
                     r"key\s+managerial\s+personnel",
                     r"at the helm",
                     r"senior management") or
        _has(p, r'chief executive officer.*chief financial officer',
             r'head.*strategy.*corporate affairs',
             r'chief executive.*prior experience')
    )),

    # ── Supply chain ──────────────────────────────────────────────────────
    ("supply_chain", 0.87, lambda p: (
        _heading_has(p, r"supply\s*chain",
                     r"manufacturing\s+park",
                     r"procurement") or
        _has(p, r'contract manufacturer|jewellery.*park|ksidc',
             r'exclusive vendor.*south india')
    )),

    # ── ESG / Sustainability / CSR ────────────────────────────────────────
    ("esg_csr", 0.88, lambda p: (
        _heading_has(p, r"esg",
                     r"environment.*social",
                     r"sustainability",
                     r"csr",
                     r"corporate social responsibility",
                     r"governance") or
        _has(p, r'csr committee|social initiative|green initiative',
             r'sustainability report|scope.*emissions')
    )),

    # ── Directors' report ─────────────────────────────────────────────────
    ("directors_report", 0.92, lambda p: (
        _heading_has(p, r"directors.?\s*report",
                     r"board.?s\s+report",
                     r"report of the directors") or
        _has(p, r"directors are pleased to present",
             r"the\s+directors\s+hereby\s+present",
             r"annual report.*company.*directors")
    )),

    # ── Corporate governance report ───────────────────────────────────────
    ("corporate_governance", 0.92, lambda p: (
        _heading_has(p, r"corporate\s+governance",
                     r"governance\s+report",
                     r"audit\s+committee",
                     r"nomination.*remuneration\s+committee",
                     r"stakeholders.*relationship\s+committee",
                     r"risk\s+management\s+committee",
                     r"board.*directors") or
        (_has(p, r'sebi.*listing.*regulation|regulation\s+\d+.*listing',
              r'independent director|non.executive.*independent',
              r'composition.*attendance.*committee') and
         _is_tabular(p))
    )),

    # ── Auditor's report (standalone) ────────────────────────────────────
    ("auditors_report_standalone", 0.93, lambda p: (
        _heading_has(p, r"independent auditor.?s report",
                     r"auditor.?s\s+report",
                     r"annexure.*auditor.?s report") and
        _has(p, r'standalone\s+financial|standalone balance sheet')
    )),

    # ── Auditor's report (consolidated) ──────────────────────────────────
    ("auditors_report_consolidated", 0.93, lambda p: (
        (_heading_has(p, r"independent auditor.?s report",
                      r"auditor.?s\s+report") and
         _has(p, r'consolidated\s+financial')) or
        _has(p, r'consolidated.*true and fair view.*auditor')
    )),

    # ── Standalone balance sheet ──────────────────────────────────────────
    ("standalone_balance_sheet", 0.95, lambda p: (
        _heading_has(p, r"standalone\s+balance\s+sheet") or
        (_is_tabular(p) and
         _has(p, r'standalone\s+balance\s+sheet',
              r'non.current assets.*property.*plant.*equipment') and
         _has_currency_figures(p))
    )),

    # ── Standalone P&L ────────────────────────────────────────────────────
    ("standalone_profit_loss", 0.95, lambda p: (
        _heading_has(p, r"standalone.*profit.*loss",
                     r"statement of profit.*loss.*standalone") or
        (_is_tabular(p) and
         _has(p, r'revenue from operations.*standalone|standalone.*revenue from operations',
              r'cost of materials consumed',
              r'profit before tax.*standalone') and
         _has_currency_figures(p))
    )),

    # ── Standalone cash flow statement ───────────────────────────────────
    ("standalone_cash_flow", 0.95, lambda p: (
        _heading_has(p, r"standalone.*cash\s+flow",
                     r"cash\s+flow.*standalone") or
        (_is_tabular(p) and
         _has(p, r'cash flow.*operating.*standalone|standalone.*operating activities',
              r'net cash.*operating|investing.*financing') and
         _has_currency_figures(p))
    )),

    # ── Standalone statement of changes in equity ─────────────────────────
    ("standalone_equity_changes", 0.93, lambda p: (
        _heading_has(p, r"standalone.*changes.*equity",
                     r"statement.*changes.*equity.*standalone") or
        (_is_tabular(p) and
         _has(p, r'standalone.*equity share capital|other equity.*standalone',
              r'changes in equity.*standalone'))
    )),

    # ── Consolidated balance sheet ────────────────────────────────────────
    ("consolidated_balance_sheet", 0.95, lambda p: (
        _heading_has(p, r"consolidated\s+balance\s+sheet") or
        (_is_tabular(p) and
         _has(p, r'consolidated\s+balance\s+sheet') and
         _has_currency_figures(p))
    )),

    # ── Consolidated P&L ──────────────────────────────────────────────────
    ("consolidated_profit_loss", 0.95, lambda p: (
        (_heading_has(p, r"consolidated.*profit.*loss",
                      r"consolidated.*statement of profit") and
         not _heading_has(p, r"consolidated\s+balance\s+sheet")) or
        (_is_tabular(p) and
         _has(p, r'consolidated.*revenue from operations',
              r'consolidated.*profit before tax',
              r'consolidated.*cost of materials') and
         not _has(p, r'consolidated\s+balance\s+sheet') and
         _has_currency_figures(p))
    )),

    # ── Consolidated cash flow ────────────────────────────────────────────
    ("consolidated_cash_flow", 0.95, lambda p: (
        _heading_has(p, r"consolidated.*cash\s+flow",
                     r"cash\s+flow.*consolidated") or
        (_is_tabular(p) and
         _has(p, r'cash flow.*operating.*consolidated|consolidated.*operating activities') and
         _has_currency_figures(p))
    )),

    # ── Consolidated statement of changes in equity ───────────────────────
    ("consolidated_equity_changes", 0.93, lambda p: (
        _heading_has(p, r"consolidated.*changes.*equity",
                     r"statement.*changes.*equity.*consolidated")
    )),

    # ── Notes to standalone financials ───────────────────────────────────
    ("notes_to_standalone_financials", 0.90, lambda p: (
        (_heading_has(p, r"^notes$", r"notes\s+forming\s+part") and
         _has(p, r'standalone financial statement',
              r'summary of material accounting polic')) or
        (_heading_has(p, r"^notes$") and
         _has(p, r'standalone') and
         _is_tabular(p) and _has_currency_figures(p))
    )),

    # ── Notes to consolidated financials ─────────────────────────────────
    ("notes_to_consolidated_financials", 0.90, lambda p: (
        (_heading_has(p, r"^notes$", r"notes\s+forming\s+part") and
         _has(p, r'consolidated financial statement')) or
        (_heading_has(p, r"^notes$") and
         _has(p, r'consolidated') and
         _is_tabular(p) and _has_currency_figures(p))
    )),

    # ── Segment information (operational / financial) ─────────────────────
    ("segment_information", 0.90, lambda p: (
        _heading_has(p, r"segment\s+(information|report|result|revenue)",
                     r"operating\s+segment") or
        (_is_tabular(p) and
         _has(p, r'reportable segment|segment.*revenue.*profit',
              r'india.*segment|international.*segment'))
    )),

    # ── Related party transactions ────────────────────────────────────────
    ("related_party_transactions", 0.92, lambda p: (
        _heading_has(p, r"related\s+party",
                     r"transactions.*related\s+party") or
        _has(p, r'key managerial personnel.*transaction|kmp.*related party',
             r'ind as 24|related party.*disclosure')
    )),

    # ── Subsidiary information ────────────────────────────────────────────
    ("subsidiary_information", 0.90, lambda p: (
        _heading_has(p, r"subsidiary",
                     r"additional information.*consolidated",
                     r"consolidation") or
        (_is_tabular(p) and
         _has(p, r'kalyan jewellers fze|enovate lifestyles|kalyan jewellers llc',
              r'net assets.*total assets.*total liab',
              r'share in profit or loss.*consolidated'))
    )),

    # ── Accounting policies ───────────────────────────────────────────────
    ("accounting_policies", 0.92, lambda p: (
        _heading_has(p, r"material accounting polic",
                     r"significant accounting polic",
                     r"summary of.*accounting") or
        _has(p, r'basis of preparation|ind as\s+\d+|going concern basis',
             r'recognition.*revenue.*accounting policy',
             r'property.*plant.*equipment.*depreciation.*accounting')
    )),

    # ── Shareholder / Investor information ───────────────────────────────
    ("shareholder_investor_info", 0.88, lambda p: (
        _heading_has(p, r"shareholder\s+focus",
                     r"investor\s+(information|relation)",
                     r"dividend",
                     r"annual general meeting") or
        (_has(p, r'dividend payout|shareholder return|agm.*date',
              r'nse.*bse.*listing|stock.*exchange.*listing') and
         _word_count(p) > 80)
    )),

    # ── Risk management / Risk factors ───────────────────────────────────
    ("risk_management", 0.88, lambda p: (
        _heading_has(p, r"risk\s+(management|factor|framework)",
                     r"key\s+risk") or
        _has(p, r'liquidity risk|credit risk|market risk|foreign exchange risk',
             r'interest rate risk|commodity.*risk|gold.*price.*risk',
             r'risk management committee')
    )),

    # ── HR / Employee information ─────────────────────────────────────────
    ("human_resources_employees", 0.85, lambda p: (
        _heading_has(p, r"human\s+resource",
                     r"employee",
                     r"staff\s+training") or
        (_has(p, r'employee benefit expense|staff training',
              r'number of employees|workforce',
              r'gratuity|provident fund') and
         _word_count(p) > 80)
    )),

    # ── Notice / AGM notice ───────────────────────────────────────────────
    ("agm_notice", 0.90, lambda p: (
        _heading_has(p, r"notice") or
        _has(p, r'notice.*annual general meeting',
             r'hereby given.*annual general meeting',
             r'special resolution.*agm')
    )),

    # ── Qualitative/narrative filler (section dividers, image-only, etc.) ─
    ("section_divider_filler", 0.80, lambda p: (
        p.is_short and
        _word_count(p) < 30 and
        not _has_currency_figures(p)
    )),

    # ── Catch-all ─────────────────────────────────────────────────────────
    ("general_narrative", 0.40, lambda p: True),
]

@dataclass
class IntentResult:
    section_name: str
    section_confidence: float

class PageIntentTagger:
    """
    Tags each page in the cleaned JSON with a page_intent label.

    """
    def __init__(self):
        pass

    def _tag_page(self, page: CleanResult) -> list[str]:
        """
        Run the intent rules against a single page dict.
        Returns the first matching IntentResult.
        """
        pn = page.page_number
        
        intent: list[str] = []
        
        for intent_label, confidence, detector in INTENT_RULES:
            if detector(page):
                intent.append(intent_label)
        return intent

        


