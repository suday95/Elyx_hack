from sentence_transformers import SentenceTransformer
import os
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np

# Initialize model globally (384-dim)
MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using local SentenceTransformer"""
    try:
        # Convert single string to list if needed
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = MODEL.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()  # Chroma expects list[list[float]]
        
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        raise