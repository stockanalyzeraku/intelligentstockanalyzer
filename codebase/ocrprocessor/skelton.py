from dataclasses import dataclass
from pathlib import Path
import re
from datetime import date
from config import CONFIG

#pattern
ALLOWED_BASE = Path(CONFIG.UPLOADS_PATH).resolve()
ALLOWED_EXTENSIONS = {".pdf", ".json"}


MAX_TEXT_LENGTH = 5_000_000

SQL_INJECTION_PATTERNS: list[str] = [

    # ── Union-based injection ────────────────────────────────────────────────
    r"(?i)\b(union\s+all\s+select)\b",          # UNION ALL SELECT
    r"(?i)\b(union\s+select)\b",                # UNION SELECT

    # ── Core DML ────────────────────────────────────────────────────────────
    r"(?i)\bselect\b.{0,300}\bfrom\b",          # SELECT ... FROM (bounded — ReDoS-safe)
    r"(?i)\b(insert\s+into)\b",                 # INSERT INTO
    r"(?i)\b(update\s+\w+\s+set)\b",            # UPDATE x SET
    r"(?i)\b(delete\s+from)\b",                 # DELETE FROM

    # ── DDL (Data Definition Language) ──────────────────────────────────────
    r"(?i)\b(drop\s+(table|database|schema|index|view|procedure|function|trigger))\b",
    r"(?i)\b(alter\s+(table|database|schema))\b",
    r"(?i)\b(create\s+(table|database|schema|index|view|procedure|function|trigger))\b",
    r"(?i)\b(truncate\s+(table\s+)?\w+)\b",     # TRUNCATE TABLE

    # ── Data exfiltration ────────────────────────────────────────────────────
    r"(?i)\b(into\s+(outfile|dumpfile))\b",     # MySQL file write
    r"(?i)\b(load\s+data\s+(local\s+)?infile)\b",  # MySQL file read
    r"(?i)\binformation_schema\b",              # schema enumeration (all DBMS)
    r"(?i)\bsysobjects\b",                      # SQL Server schema table
    r"(?i)\bsyscolumns\b",                      # SQL Server column table
    r"(?i)\bpg_catalog\b",                      # PostgreSQL schema
    r"(?i)\ball_tables\b",                      # Oracle schema

    # ── Stored procedure / extended proc execution ───────────────────────────
    r"(?i)exec(\s|\()(xp_|sp_)",               # SQL Server xp_cmdshell, sp_*
    r"(?i)\bexecute\s+(immediate|sp_|xp_)",    # Oracle EXECUTE IMMEDIATE
    r"(?i)\bcall\s+\w+\s*\(",                  # MySQL/PostgreSQL CALL proc()

    # ── Comment-based evasion ─────────────────────────────────────────────────
    r"(?i)(--|\#)\s*$",                         # line-ending comment
    r"(?i)(;\s*--)",                            # statement + comment
    r"/\*[\s\S]{0,500}?\*/",                   # block comment (bounded)

    # ── Boolean-based blind injection ────────────────────────────────────────
    r"(?i)(\bor\b\s+\d+\s*=\s*\d+)",           # OR 1=1, OR 2=2, etc.
    r"(?i)(\band\b\s+\d+\s*=\s*\d+)",          # AND 1=1
    r"'\s*or\s*'.*?'\s*=\s*'",                 # 'x' OR 'a'='a'
    r"(?i)(\bor\b\s+['\"].*?['\"]\s*=\s*['\"])",  # OR 'abc'='abc'
    r"(?i)\bhaving\s+\d+\s*=\s*\d+",           # HAVING 1=1 (error-based)

    # ── Time-based blind injection ────────────────────────────────────────────
    r"(?i)(waitfor\s+delay)",                   # SQL Server
    r"(?i)(benchmark\s*\()",                    # MySQL
    r"(?i)(sleep\s*\(\s*\d+\s*\))",            # MySQL
    r"(?i)(pg_sleep\s*\(\s*\d+\s*\))",         # PostgreSQL
    r"(?i)(dbms_pipe\.receive_message)",         # Oracle

    # ── Obfuscation and encoding ─────────────────────────────────────────────
    r"(?i)\bchar\s*\(\s*\d+",                  # CHAR(65) obfuscation
    r"(?i)\bconcat\s*\(.{0,100}select",         # CONCAT() wrapping SELECT
    r"(?i)\bgroup_concat\s*\(",                 # MySQL data exfil function
    r"(?i)0x[0-9a-fA-F]{4,}",                  # hex-encoded string/payload
    r"(?i)\bcast\s*\(.{0,50}as\s+(text|varchar|char|nvarchar)\b",  # CAST() bypass

    # ── Stacked / batched queries ─────────────────────────────────────────────
    r"(?i);\s*(select|insert|update|delete|drop|alter|create|truncate|exec|call)\b",

    # ── Privilege manipulation ────────────────────────────────────────────────
    r"(?i)\b(grant|revoke)\s+\w+",             # privilege escalation
    r"(?i)\bwith\s+grant\s+option\b",          # privilege propagation
]

CODE_INJECTION_PATTERNS: list[str] = [

    # ── XSS / HTML tag injection ─────────────────────────────────────────────
    r"(?i)<script[\s>]",
    r"(?i)</script>",
    r"(?i)javascript:",
    r"(?i)vbscript:",                            # IE-specific JS scheme
    r"(?i)on\w+\s*=\s*['\"]",                   # inline event handlers (onclick, onerror, etc.)
    r"(?i)<iframe[\s>]",
    r"(?i)<object[\s>]",
    r"(?i)<embed[\s>]",
    r"(?i)<form[\s>]",
    r"(?i)<base[\s>]",
    r"(?i)<link[\s>]",
    r"(?i)<style[\s>]",
    r"(?i)<meta[^>]*http-equiv\s*=\s*['\"]?refresh",
    r"(?i)expression\s*\(",                     # CSS expression() — IE XSS

    # ── Server-Side Template Injection (SSTI) ─────────────────────────────────
    r"(?i)\{\{.{0,200}?\}\}",                  # Jinja2/Twig/Vue: {{ 7*7 }}
    r"(?i)\$\{.{0,200}?\}",                    # JS template literals / Java EL
    r"(?i)\$\{T\s*\(",                          # Spring SpEL: ${T(java.lang.Runtime)...}
    r"(?i)#\{.{0,200}?\}",                     # JSF/Ruby EL: #{expression}
    r"(?i)<%=?.{0,200}?%>",                    # JSP / ERB / EJS
    r"(?i)\{%.{0,200}?%\}",                    # Jinja2 block: {% for x in y %}
    r"(?i)\{#.{0,200}?#\}",                    # Jinja2 comment (can hide injection)

    # ── Python dynamic code execution ─────────────────────────────────────────
    r"(?i)\beval\s*\(",
    r"(?i)\bexec\s*\(",
    r"(?i)\bcompile\s*\(",                      # compile(src, '<string>', 'exec')
    r"(?i)\b__import__\s*\(",
    r"(?i)\bimportlib\.import_module\s*\(",

    # ── OS / subprocess access ────────────────────────────────────────────────
    r"(?i)\bsubprocess\.",
    r"(?i)\bos\.system\s*\(",
    r"(?i)\bos\.popen\s*\(",
    r"(?i)\bos\.spawn[lve]?[lpe]?\s*\(",        # os.spawnl, os.spawnve, etc.
    r"(?i)\bos\.exec[vle]?[pe]?\s*\(",          # os.execv, os.execlpe, etc.

    # ── Dangerous deserialization ─────────────────────────────────────────────
    r"(?i)\bpickle\.(load|loads)\s*\(",         # arbitrary code on deserialise
    r"(?i)\bmarshal\.(load|loads)\s*\(",        # bytecode deserialise
    r"(?i)\byaml\.load\s*\(",                   # unsafe yaml.load (not safe_load)
    r"(?i)\bjsonpickle\.decode\s*\(",           # arbitrary object decode

    # ── Python introspection / sandbox escape ────────────────────────────────
    r"(?i)\bglobals\s*\(\s*\)",
    r"(?i)\blocals\s*\(\s*\)",
    r"(?i)\bvars\s*\(\s*\)",
    r"(?i)\bdir\s*\(\s*\)",
    r"(?i)__builtins__",
    r"(?i)__globals__",
    r"(?i)__class__",
    r"(?i)__subclasses__\s*\(\s*\)",
    r"(?i)__bases__",
    r"(?i)__mro__",
    r"(?i)\bgetattr\s*\(.{0,100}['\"]__",      # getattr(obj, '__secret__')
    r"(?i)\bsetattr\s*\(",
    r"(?i)\bdelattr\s*\(",

    # ── Native / low-level access ─────────────────────────────────────────────
    r"(?i)\bctypes\.",                          # direct memory / C ABI access
    r"(?i)\bshutil\.",                          # bulk file system operations

    # ── File write in unexpected mode ─────────────────────────────────────────
    r"(?i)\bopen\s*\(.{0,200}['\"][wa+]",      # open() in write or append mode

    # ── Shell metacharacter injection ─────────────────────────────────────────
    r"`[^`]{0,300}`",                           # backtick execution: `rm -rf /`
    r"(?i)\$\([^)]{0,300}\)",                  # $(command) shell expansion
]

PATH_TRAVERSAL_PATTERN = re.compile(
    r"(\.\./|\.\.\\|"           # literal
    r"%2e%2e[%/\\]|"            # single URL-encoded
    r"%252e%252e[%/\\]|"        # double URL-encoded
    r"\.\.[/\\]|"               # dot-dot with sep
    r"\x00)",                   # null byte
    re.IGNORECASE
)
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
PRIVATE_IP_PATTERN = re.compile(
    r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|"   # RFC1918
    r"169\.254\.|"                                       # link-local
    r"fe[89ab][0-9a-f]:|fc[0-9a-f]{2}:|fd)",            # IPv6 private
    re.IGNORECASE
)
PRIVATE_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",          # cloud metadata (AWS/Azure/GCP)
    "metadata.google.internal", # GCP alt
    "metadata.goog",            # GCP alt
}

_SCRIPT_NAME_RE = re.compile(r"^[A-Z0-9_\-]{1,64}$")
NESTING_THRESHOLD = 8

MIN_YEAR = 1995
MAX_YEAR = date.today().year + 1


#classes
@dataclass
class PageContent:
    """Raw OCR content extracted from one page of the source PDF."""
    page_number: int
    text: str
