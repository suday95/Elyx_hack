import pandas as pd
import chromadb
import yaml
import os
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
from chromadb.config import Settings
from rag.utils.text import embed  # This now uses Gemini embeddings
from rag.utils.io import load_csv
import time  # For rate limiting

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

chroma_path = project_root / "chroma"
chroma_path.mkdir(parents=True, exist_ok=True)

# Initialize Chroma client
client = chromadb.PersistentClient(
    path=str(chroma_path),
    settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True
    )
)

# Delete existing collection to recreate with correct settings
try:
    client.delete_collection("elyx_docs")
    print("Deleted existing collection to reset with new embedding settings.")
except:
    print("Creating new collection")

# Create collection without embedding function since we're providing our own embeddings
collection = client.create_collection(
    name="elyx_docs",
    metadata={
        "hnsw:space": "cosine",
        "embedding_dimensions": 384,
    } 
)

def process_row(row, data_type):
    """Convert CSV row to document format with type-specific handling"""
    # Create doc_id based on date or month field
    if data_type == "kpi":
        doc_id = f"{data_type}:{row['month']}"
    else:
        doc_id = f"{data_type}:{row['date']}"
    
    metadata = {"type": data_type, "doc_id": doc_id}
    text_fields = []

    # Type-specific processing
    if data_type == "lab":
        # Handle labs data
        metadata.update({
            "date": row["date"],
            "ldl": float(row["ldl_mgdl"]),
            "apob": float(row["apob_mgdl"])
        })
        text_fields = [
            f"ldl:{row['ldl_mgdl']}", f"apob:{row['apob_mgdl']}",
            f"hdl:{row['hdl_mgdl']}", f"triglycerides:{row['triglycerides_mgdl']}"
        ]

    elif data_type == "daily":
        metadata.update({
            "date": row["date"],
            "rhr": int(row["rhr_bpm"]),
            "hrv": float(row["hrv_ms"])
        })
        text_fields = [
            f"steps:{row['steps']}", f"rhr:{row['rhr_bpm']}",
            f"hrv:{row['hrv_ms']}", f"sleep:{row['sleep_hours']}h"
        ]

    elif data_type == "body_comp":
        metadata.update({
            "date": row["date"],
            "bodyfat": float(row["dexa_bodyfat_percent"])
        })
        text_fields = [
            f"bodyfat:{row['dexa_bodyfat_percent']}%",
            f"lean_mass:{row['dexa_lean_mass_kg']}kg",
            f"bone_density:{row['bone_density_tscore']}"
        ]

    elif data_type == "fitness":
        metadata.update({
            "date": row["date"],
            "vo2max": float(row["vo2max_est"])
        })
        text_fields = [
            f"vo2max:{row['vo2max_est']}", 
            f"deadlift:{row['1rm_deadlift_kg']}kg",
            f"squat:{row['1rm_squat_kg']}kg"
        ]

    elif data_type == "intervention":
        metadata.update({
            "date": row["date"],
            "rule_id": row["rule_id"],
            "owner": row["owner"]
        })
        text_fields = [
            f"trigger:{row['trigger_metric']}={row['trigger_value']}",
            f"action:{row['action']}",
            f"owner:{row['owner']}"
        ]

    elif data_type == "kpi":
        metadata.update({
            "month": row["month"],
            "adherence": float(row["adherence_avg"])
        })
        text_fields = [
            f"adherence:{row['adherence_avg']}",
            f"sessions:{row['sessions_total']}",
            f"weight_change:{row['weight_change_kg']}kg"
        ]

    elif data_type == "event":
        metadata.update({
            "date": row["date"],
            "event_type": row["event_type"]
        })
        text_fields = [
            f"event:{row['event_type']}",
            f"intensity:{row['intensity']}",
            f"notes:{row['notes'][:50]}..."  # Truncate long notes
        ]

    # Convert all values to strings for text field
    text_fields = [str(field) for field in text_fields]
    
    return {
        "id": doc_id,
        "text": " | ".join(text_fields),
        "metadata": metadata
    }

def ingest_data():
    docs = []
    
    # Load profile from YAML
    try:
        with open('config/profile.yaml') as f:
            profile = yaml.safe_load(f)
            profile_text = " | ".join([
                f"name:{profile['name']}",
                f"age:{profile['age']}",
                f"sex:{profile['sex']}",
                f"goals:{', '.join(profile['goals'])}"
            ])
            docs.append({
                "id": f"profile:{profile['member_id']}",
                "text": profile_text,
                "metadata": {
                    "type": "profile",
                    "doc_id": f"profile:{profile['member_id']}"
                }
            })
        print("Processed profile.yaml")
    except Exception as e:
        print(f"Error loading profile: {str(e)}")

    # Process all CSV files
    csv_mapping = [
        ("body_comp", "data/body_comp.csv"),
        ("daily", "data/daily.csv"),
        ("event", "data/events.csv"),
        ("fitness", "data/fitness.csv"),
        ("intervention", "data/interventions.csv"),
        ("kpi", "data/kpis_monthly.csv"),
        ("lab", "data/labs_quarterly.csv")
    ]
    
    for data_type, path in csv_mapping:
        try:
            df = pd.read_csv(path)
            # Convert date columns to string format
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            for _, row in df.iterrows():
                # Replace NaN values with empty strings
                row = row.replace({np.nan: None})
                docs.append(process_row(row, data_type))
            print(f"Processed {path} with {len(df)} rows")
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Batch upsert with Gemini embeddings
    batch_size = 50  # Reduced for Gemini API rate limits
    total_docs = len(docs)
    print(f"Total documents to ingest: {total_docs}")
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i:i+batch_size]
        try:
            # Get texts for embedding
            texts = [doc["text"] for doc in batch]
            
            # Generate embeddings with Gemini (API call)
            embeddings = embed(texts)
            
            # Prepare data for Chroma
            ids = [doc["id"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            documents = [doc["text"] for doc in batch]
            
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"Upserted batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
            
            # Add delay to avoid rate limiting (Gemini has 60 RPM free tier)
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Failed to upsert batch {i//batch_size + 1}: {str(e)}")
            # Add extra delay on error
            time.sleep(5)

    print(f"Ingestion complete. Total documents: {collection.count()}")

if __name__ == "__main__":
    ingest_data()
    print("\n=== Storage Verification ===")
    print(f"Collection count: {collection.count()}")
    print(f"Storage path: {os.path.abspath('chroma')}")

    # New way to verify persistence
    try:
        # Create new client to verify independent connection
        test_client = chromadb.PersistentClient(path=str(chroma_path))
        test_col = test_client.get_collection("elyx_docs")
        print(f"Verified document count: {test_col.count()}")
        print("Sample document:", test_col.peek(limit=1))
    except Exception as e:
        print(f"Verification failed: {str(e)}")