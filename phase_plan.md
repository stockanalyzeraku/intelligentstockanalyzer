Phase 1 — The Working Brain
Goal: Process PDFs for one company, query reliably, get genuinely useful answers. Nothing else.
Duration: 3-4 weeks
Who can use it: Only you
What Is Included
INFRASTRUCTURE:
├── GitHub repository set up
├── Google Drive folder structure finalised
├── Config class with all settings
├── Environment variable management
└── Session restart script (5-minute cold start)

THE BRAIN
├── PDF quality classifier
├── Section boundary detector
├── Hierarchical chunker (parent + child)
├── Table extractor with structure
├── Financial fact extractor
├── Deduplication layer
├── Four ChromaDB collections
├── Local embedding (all-MiniLM-L6-v2)
├── Hybrid search (semantic + BM25)
├── Query classification
├── Query decomposition and expansion
├── REACT agent loop
├── Gap analyser
├── Synthesis engine
└── Working memory

ROBUSTNESS (Phase 1 level):
├── Structured logger
│   File-based, JSON format, to Drive
│   INFO/WARNING/ERROR/CRITICAL levels
├── Basic error classification
│   Transient vs permanent errors
│   Retry for transient (3 attempts, backoff)
│   Fail fast for permanent
├── Input validation
│   Question length limits
│   Scrip existence check
│   PDF format validation
└── Timeouts on all external calls
    LLM: 30 seconds
    ChromaDB: 10 seconds

DATA LAYER:
├── SQLite for metadata
│   Processed files tracking
│   Query history
│   Chunk statistics
└── All data on Drive (persists)

TESTING:
├── Manual test script
│   10 known questions with expected answers
│   Run after every code change
└── Basic assertion checks on chunk counts
What Is Not Included In Phase 1
├── Multiple companies (one company only)
├── News pipeline
├── Financial data API
├── Any user interface
├── Authentication
├── Concurrency handling
└── Monitoring
Phase 1 Success Criteria
Before moving to Phase 2, these must all pass:

□ 7 PDFs processed without errors
□ Chunk counts in expected range per PDF
□ 10 test questions all produce cited answers
□ Zero silent failures (every error logged)
□ Session restart takes < 5 minutes
□ Query response time < 30 seconds
□ No hallucinated numbers in 20 test queries
□ Log file shows clear trail for every query

Phase 2 — Multi-Company + Internal Tool
Goal: Expand to 5 companies, add news and financial data, build a usable internal interface. Show this to 5-10 trusted people and get feedback.
Duration: 4-6 weeks
Who can use it: You + trusted testers (manually share Streamlit URL via ngrok)
What Is Included
DATA EXPANSION:
├── Pipeline runs for all 5 companies
├── News ingestion pipeline
│   NewsAPI integration
│   Sentiment classification
│   Embedded into news_chunks collection
├── Financial data layer
│   BSE XBRL parsing
│   Manual entry fallback
│   Stored in SQLite
└── Report generation agent
    Runs all 8 seed prompts per company
    Stores sections in SQLite
    Regenerates when new PDF processed

ROBUSTNESS ADDITIONS:
├── Circuit breaker pattern
│   For LLM API calls
│   For ChromaDB operations
│   Fail open (degraded mode) not fail closed
├── LLM response parser hardening
│   Multiple format fallbacks
│   Never fails silently on parse error
│   Falls back to raw response if structured fails
├── Concurrency protection
│   Thread locks on BM25 index
│   Thread locks on ChromaDB writes
│   Queue for PDF processing (one at a time)
├── Memory bounds
│   BM25 index max 50,000 documents
│   LRU cache for embeddings
│   Explicit cleanup after PDF processing
└── Health check function
    Checks all components before accepting queries
    Returns status dict, logs issues

INTERNAL STREAMLIT UI:
├── Company selector
├── Living report display
│   All sections with timestamps
│   News feed with sentiment colours
├── Query interface
│   Chat-style input
│   Answer with citations
│   Follow-up question buttons
│   Conversation history in session
└── Admin panel (simple)
    Trigger PDF processing
    View processing logs
    Check chunk counts per company

PERSISTENCE IMPROVEMENTS:
├── SQLite schema hardened
│   All Phase 1 tables plus report_sections
│   Soft deletes (never hard delete)
│   Timestamps on everything
└── Drive backup script
    Weekly SQLite backup
    Alert if Drive space < 2GB
What Is Not Included In Phase 2
├── Real user authentication
├── Railway deployment
├── Payment system
├── Production database (still SQLite)
└── Automated scheduling (still manual triggers)
Phase 2 Success Criteria
□ All 5 companies processed and queryable
□ News pipeline runs without errors
□ Report generated for all companies
□ Streamlit UI usable by non-technical person
│  (test: give to someone unfamiliar, watch them use it)
□ 5 trusted people have used it and given feedback
□ At least one person says: "I would pay for this"
□ Response time < 15 seconds for 95% of queries
□ Zero crashes during tester sessions
□ All errors logged with enough context to debug

Phase 3 — Production Deployment
Goal: Move off Colab onto Railway. Persistent server, real URL, no session resets. First paying customers.
Duration: 6-8 weeks
Who can use it: First paying customers (target: 10-20)
The Colab To Railway Migration
THIS IS SIMPLER THAN IT SOUNDS.
Because we used abstractions from the start.

WHAT CHANGES:
├── Config.CHROMA_PATH: Drive path → Railway volume path
├── Config.SQLITE_DB_PATH: Drive path → Railway volume path
├── Config.UPLOAD_PATH: Drive path → Railway volume path
└── API keys: os.environ (same approach, different values)

WHAT DOES NOT CHANGE:
├── All Python code (zero changes)
├── All ChromaDB operations (same client API)
├── All LLM calls (same Gemini client)
├── All embedding operations (same model)
└── All agent logic (completely unchanged)

MIGRATION STEPS:
1. Export ChromaDB from Drive (zip the folder)
2. Export SQLite database (copy the file)
3. Deploy to Railway (push GitHub repo)
4. Upload ChromaDB zip to Railway volume
5. Upload SQLite to Railway volume
6. Set environment variables in Railway dashboard
7. Test all 5 companies work
8. Point domain to Railway URL
What Is Included In Phase 3
INFRASTRUCTURE:
├── Railway deployment
│   FastAPI backend (replaces notebook API calls)
│   Gunicorn worker for concurrent requests
│   Persistent volume for ChromaDB + SQLite
│   Environment variables in Railway dashboard
├── Vercel deployment
│   Next.js frontend (replaces Streamlit)
│   Connects to Railway FastAPI backend
├── Domain setup
│   Custom domain on Vercel
│   SSL certificate (automatic)
└── GitHub Actions CI/CD
    Push to main → auto deploy to Railway
    Push to main → auto deploy to Vercel
    Run test suite before deploying

AUTHENTICATION:
├── Supabase Auth integration
│   Email + password
│   Google OAuth
│   JWT tokens
├── Protected routes
│   Every API endpoint requires valid JWT
│   Frontend redirects to login if not authenticated
└── Role system
    Free tier: 5 companies, 10 queries/day
    Paid tier: all companies, unlimited
    Admin: pipeline management

DATABASE UPGRADE:
├── SQLite → PostgreSQL on Railway
│   SQLAlchemy models unchanged
│   Just change connection string
│   Migrate data with one script
├── Add missing tables
│   users, sessions, subscriptions
│   rate_limit_tracking
└── Add all indexes from schema design

RATE LIMITING:
├── Per user per endpoint
├── Redis-based sliding window
│   Free: 10 queries/hour
│   Paid: 100 queries/hour
└── Returns 429 with retry-after header

CACHING LAYER:
├── Redis on Railway
├── Report sections cached 6 hours
├── Query results cached 1 hour
├── Cache invalidated on new PDF processed
└── Cache warmed on deployment

TASK QUEUE:
├── ARQ + Redis
├── PDF processing as background job
│   API returns job_id immediately
│   Frontend polls for completion
├── Report generation as background job
└── News fetching scheduled daily

OBSERVABILITY:
├── Langfuse integration
│   Every LLM call logged
│   Prompt + response + latency + tokens
├── Sentry integration
│   All unhandled exceptions captured
│   Alert on first occurrence of new error
├── Structured logging to Railway logs
│   Searchable in Railway dashboard
└── Basic Grafana dashboard
    Query volume, error rate, latency

ROBUSTNESS ADDITIONS:
├── Prompt injection detection middleware
├── Request ID propagation (distributed tracing)
├── Graceful shutdown handling
│   Finish in-progress requests before stopping
│   Do not accept new requests during shutdown
├── Memory usage monitoring
│   Alert if > 80% of Railway instance memory
└── Automated ChromaDB backup
    Daily backup to Cloudflare R2
Phase 3 Success Criteria
□ System runs 24/7 without Colab
□ 10 paying customers using it
□ P95 query latency < 10 seconds
□ Zero data loss incidents
□ Error rate < 2% of all requests
□ Authentication working correctly
□ Rate limiting working correctly
□ First month revenue > infrastructure cost
□ You can go to sleep without worrying about it

Phase 4 — Scale To 100 Users
Goal: Harden everything, improve quality, reach 100 paying users.
Duration: 8-12 weeks (ongoing)
Who can use it: 100 paying customers
What Is Included
DATABASE:
├── SQLite → PostgreSQL (done in Phase 3)
├── Add read replica for query-heavy load
├── Connection pooling (PgBouncer)
└── Query performance monitoring

VECTOR DATABASE:
├── ChromaDB → Qdrant Cloud
│   Same query API, just different client
│   Better performance at this scale
│   Managed backups
│   No maintenance overhead
└── Existing data migrated with one script

EMBEDDING UPGRADE:
├── all-MiniLM-L6-v2 → bge-large-en-v1.5
│   Better quality, still local
│   Re-embed all existing chunks (one-time job)
└── Embedding service as separate Railway process
    Dedicated CPU, not shared with API

QUALITY IMPROVEMENTS:
├── Cross-encoder re-ranking added
│   Top 20 chunks → re-ranked → top 8 sent to LLM
│   Measurable improvement in answer quality
├── Evaluation dataset built
│   100 questions with ground truth answers
│   Run weekly, track quality metrics over time
├── Dynamic prompt generation
│   Agent generates its own follow-up prompts
│   Discovers insights beyond seed prompts
└── Answer quality scoring
    Every answer scored after generation
    Low scores flagged for review

FEATURES:
├── Valuation scenario engine
│   Bear/base/bull case per company
│   Reverse DCF calculator
├── Company comparison queries
│   "Compare Kalyan vs Titan on digital"
│   Multi-scrip retrieval
├── Trend alerts
│   User sets a topic to watch
│   System alerts when new filing mentions it
└── Export to PDF
    Download any report as formatted PDF

BUSINESS:
├── Subscription management (Stripe)
├── Usage dashboard for users
├── Email notifications for new reports
└── Referral system

The Detailed Week-by-Week For Phase 1
Since Phase 1 is where you are starting, here is the exact sequence:
WEEK 1:
Day 1-2: GitHub setup, folder structure, Config class
Day 3-4: Logger class, error classification system
Day 5-7: Session restart script, test on clean Colab

WEEK 2:
Day 1-3: Fix and harden the brain code we built
          (it has gaps we identified — fix them)
Day 4-5: Input validation layer
Day 6-7: Timeouts on all external calls

WEEK 3:
Day 1-3: SQLite layer (replace ad-hoc db.py)
Day 4-5: Retry logic with exponential backoff
Day 6-7: Manual test suite (10 questions, known answers)

WEEK 4:
Day 1-3: Run all 7 PDFs through pipeline
          Fix every error that appears
Day 4-5: Run test suite, fix failures
Day 6-7: Document what works, what does not
          Decide if Phase 1 criteria are met
