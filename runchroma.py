from codebase.vectordb.chromastore import ChromaStore
from config import CONFIG
import os
store = ChromaStore.get_instance()

# Store
path = os.path.join(CONFIG.UPLOADS_PATH, "KALYANKJIL", "ANNUAL_2025", "KALYAN_ANNUAL_EMBEDDINGREADY_2025.json")
#store.store_in_chromadb(path, "kalyan_annual_2025")

# Query (your agent will use this)
results = store.query_collection(
    collection_name = "kalyan_annual_2025",
    query_texts     = ["what is the revenue for FY25?"],
    n_results       = 5,
    where           = {"company": "KALYANKJIL"},   # optional filter
)

# # Get by ID
# chunk = store.get_by_id("kalyan_annual_2025", "KALYANKJIL_2025_financials_14_0")

# Status
print(store.status("kalyan_annual_2025"))
# ── Print cleanly ─────────────────────────────────────────────────
ids       = results["ids"][0]
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, (doc_id, text, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
    print(f"{'='*60}")
    print(f"Result {i+1}")
    print(f"ID         : {doc_id}")
    print(f"Distance   : {dist:.4f}")
    print(f"Metadata   : {meta}")
    print(f"Text       : {text[:300]}")
    print()