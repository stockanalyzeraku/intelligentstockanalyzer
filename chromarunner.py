from codebase.vectordb.chromastore import ChromaStore
from config import CONFIG
import os
store = ChromaStore.get_instance()

# Store
# path = os.path.join(CONFIG.UPLOADS_PATH, "KALYANKJIL", "ANNUAL_2025", "KALYANKJIL_ANNUAL_2025_EMBEDDINGREADY.json")
# store.store_in_chromadb(path, "kalyan_annual")
# print("Store 2025")
# path = os.path.join(CONFIG.UPLOADS_PATH, "KALYANKJIL", "ANNUAL_2024", "KALYANKJIL_ANNUAL_2024_EMBEDDINGREADY.json")
# store.store_in_chromadb(path, "kalyan_annual")
# print("stored 2024")
path = os.path.join(CONFIG.UPLOADS_PATH, "KALYANKJIL", "ANNUAL_2023", "KALYANKJIL_ANNUAL_2023_EMBEDDINGREADY.json")
store.store_in_chromadb(path, "kalyan_annual")
print("stored 2023")

# Query child collection, then expand to parent pages
results = store.query_children_with_parent_context(
    query_texts     = ["what is the revenue for FY23?"],
    n_results       = 5,
    where           = {"company": "KALYANKJIL"},   # optional filter
)

print(store.status(CONFIG.COL_CHILD))
print(store.status(CONFIG.COL_PARENT))

for i, result in enumerate(results):
    print(f"{'='*60}")
    print(f"Result {i+1}")
    print(f"Parent ID  : {result['parent_id']}")
    print(f"Child ID   : {result['child_id']}")
    print(f"Distance   : {result['distance']:.4f}")
    print(f"Parent meta: {result['parent_metadata']}")
    print(f"Child meta : {result['child_metadata']}")
    print(f"Parent text: {result['parent_text'][:300]}")
    print()