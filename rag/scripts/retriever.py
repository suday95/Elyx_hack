from rag.utils.text import embed
import chromadb
from pathlib import Path
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Any, Optional
# Chroma path (must match ingestion path)
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
# Initialize model at startup (not per-request)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
CHROMA_PATH = Path(__file__).parent.parent / "chroma"
from rag.scripts.router import route
# Initialize client
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
    
)


# Access collection
try:
    collection = client.get_collection("elyx_docs")
except Exception as e:
    print(f"Collection error: {str(e)}")
    print("Available collections:", [col.name for col in client.list_collections()])
    raise

# # 2. Initialize client with error handling
# def get_chroma_collection():
#     try:
#         client = chromadb.PersistentClient(
#             path=str(CHROMA_PATH),
#             settings=Settings(anonymized_telemetry=False)
#         )
        
#         # Verify collection exists
#         collections = client.list_collections()
#         if not any(col.name == "elyx_docs" for col in collections):
#             raise ValueError(f"No 'elyx_docs' collection found. Available: {[col.name for col in collections]}")
            
#         return client.get_collection("elyx_docs")
    
#     except Exception as e:
#         print(f"Error accessing ChromaDB: {str(e)}")
#         print(f"Storage path contents: {os.listdir(CHROMA_PATH) if CHROMA_PATH.exists() else 'Missing directory'}")
#         raise


ROLE_FILTERS = {
    "Ruby": {"type": {"$in": ["event", "intervention", "chat", "daily", "fitness", "body_comp", "event", "intervention"]}},
    "Dr. Warren": {"type": {"$in": ["lab", "intervention", "chat"]}},
    "Advik": {"type": {"$in": ["daily", "fitness", "chat"]}},
    "Carla": {"type": {"$in": ["daily", "body_comp", "chat"]}},
    "Rachel": {"type": {"$in": ["fitness", "body_comp", "chat"]}},
    "Neel": {"type": {"$in": ["kpi", "intervention", "chat"]}},
}
def to_ts(dstr: str) -> int:
    return int(datetime.fromisoformat(dstr).replace(tzinfo=timezone.utc).timestamp())
def normalize_role(role: str) -> str:
    role = role.strip().title()
    if role in ROLE_FILTERS:
        return role
    return "Ruby"  # Default to Auto for unknown roles
def _build_where(role_where: Dict[str, Any], since_iso: Optional[str]) -> Dict[str, Any]:
    # Build a Chroma filter that uses a single operator when combining conditions
    # Cases:
    # 1) only role_where
    # 2) only date filter
    # 3) both -> use $and
    if role_where and since_iso:
        return {"$and": [role_where, {"date": {"$gte": since_iso}}]}
    elif role_where:
        return role_where
    elif since_iso:
        return {"date": {"$gte": since_iso}}
    else:
        return {}  # no filter
def _normalize_date(since: Optional[str]) -> Optional[str]:
    if not since:
        return None
    try:
        # Accept 'YYYY-MM-DD' or full ISO, return 'YYYY-MM-DD'
        return datetime.fromisoformat(since).date().isoformat()
    except Exception:
        # Assume caller gave ISO date already; you may still want to validate
        return since

def retrieve(query, role=None, k=3, since=None):
    if(role ==None):
        role = route(query)
    normalized_role = normalize_role(role)
    role_ = ROLE_FILTERS.get(normalized_role, {}).copy()
    if since:
        since = str(since)
        since_iso = _normalize_date(since)
        since_ts = to_ts(since_iso)
    where = _build_where(role_, None)

    # if since:
    #     where["date"] = {"$gte": since}
    
    results = collection.query(
        query_embeddings=[embed([query])[0]],
        n_results=k,
        where=where
    )
    
    return [
        {
            "text": results["documents"][0][i][:300],
            "metadata": results["metadatas"][0][i],
            "id": results["ids"][0][i]
        }
        for i in range(len(results["ids"][0]))
    ]
