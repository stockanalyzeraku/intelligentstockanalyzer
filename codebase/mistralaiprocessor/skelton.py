from dataclasses import dataclass
from pathlib import Path
from config import CONFIG
#pattern
ALLOWED_BASE = Path(CONFIG.UPLOADS_PATH).resolve()
ALLOWED_EXTENSIONS = {".pdf", ".json"}


MAX_TEXT_LENGTH = 5_000_000

SQL_INJECTION_PATTERNS = [
    r"(?i)\b(union\s+select)\b",
    r"(?i)\b(select\s+.*\s+from)\b",
    r"(?i)\b(insert\s+into)\b",
    r"(?i)\b(update\s+\w+\s+set)\b",
    r"(?i)\b(delete\s+from)\b",
    r"(?i)\b(drop\s+(table|database))\b",
    r"(?i)\b(alter\s+table)\b",
    r"(?i)exec(\s|\()(xp_|sp_)",
    r"(?i)(--|\#)\s*$",
    r"(?i)(;\s*--)",
    r"(?i)(\bor\b\s+1\s*=\s*1)",
    r"(?i)(\band\b\s+1\s*=\s*1)",
    r"'\s*or\s*'.*?'\s*=\s*'",
    r"(?i)(waitfor\s+delay)",
    r"(?i)(benchmark\s*\()",
    r"(?i)(sleep\s*\(\s*\d+\s*\))",
]

CODE_INJECTION_PATTERNS = [
    r"(?i)<script[\s>]",
    r"(?i)</script>",
    r"(?i)javascript:",
    r"(?i)on\w+\s*=\s*['\"]",
    r"(?i)<iframe[\s>]",
    r"(?i)<object[\s>]",
    r"(?i)<embed[\s>]",
    r"(?i)<form[\s>]",
    r"(?i)<base[\s>]",
    r"(?i)<link[\s>]",
    r"(?i)<style[\s>]",
    r"(?i)<meta[^>]*http-equiv\s*=\s*['\"]?refresh",
    r"(?i)expression\s*\(",
    r"(?i)\{\{.*?\}\}",
    r"(?i)\$\{.*?\}",
    r"(?i)<%.*?%>",
    r"(?i)eval\s*\(",
    r"(?i)exec\s*\(",
    r"(?i)__import__\s*\(",
    r"(?i)subprocess\.",
    r"(?i)os\.system\(",
]

PATH_TRAVERSAL_PATTERN = r"(\.\./|\.\.\\)"
DANGEROUS_DATA_URI_PATTERN = r"(?i)data:(text/html|application/javascript|application/x-sh)"

DANGEROUS_FORMAT_CHARS = {
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}

# Zero-width/invisible chars — legitimate in some scripts (e.g. ZWJ/ZWNJ in
# Arabic/Hindi/Punjabi shaping), so not hard-blocked, but stripped before
# pattern-matching (to stop keyword-splitting evasion) and checked for
# abnormal runs (never legitimate).
ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff\u00ad"

MARKDOWN_URL_PATTERN = r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)"
DANGEROUS_URI_SCHEMES = {"javascript", "vbscript", "file", "data", "about", "chrome", "jar"}
PRIVATE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"}
PRIVATE_IP_PATTERN = r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)"

NESTING_THRESHOLD = 25

MIN_YEAR = 1995
MAX_YEAR = 2026


#classes
@dataclass
class PageContent:
    """Raw OCR content extracted from one page of the source PDF."""
    page_number: int
    text: str
