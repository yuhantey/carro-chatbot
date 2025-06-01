import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import pickle

MODEL_CACHE_DIR = "model_cache"
EMBEDDINGS_FILE = os.path.join(MODEL_CACHE_DIR, "embeddings.pkl")
INDEX_FILE = os.path.join(MODEL_CACHE_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(MODEL_CACHE_DIR, "chunks.pkl")

# def save_model_artifacts(model, embeddings, index, chunks):
#     """Save model artifacts to disk"""
#     os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
#     model.save(MODEL_CACHE_DIR)
    
#     with open(EMBEDDINGS_FILE, 'wb') as f:
#         pickle.dump(embeddings, f)
    
#     faiss.write_index(index, INDEX_FILE)
    
#     with open(CHUNKS_FILE, 'wb') as f:
#         pickle.dump(chunks, f)

# def load_model_artifacts():
#     """Load model artifacts from disk"""
#     if not os.path.exists(MODEL_CACHE_DIR):
#         return None, None, None, None
    
#     try:
#         model = SentenceTransformer(MODEL_CACHE_DIR)
#         with open(EMBEDDINGS_FILE, 'rb') as f:
#             embeddings = pickle.load(f)
        
#         index = faiss.read_index(INDEX_FILE)
        
#         with open(CHUNKS_FILE, 'rb') as f:
#             chunks = pickle.load(f)
        
#         return model, embeddings, index, chunks
#     except Exception as e:
#         print(f"Error loading cached model: {e}")
#         return None, None, None, None

@st.cache_resource
def build_faq_index(chunks: list[str]):
    """Build FAQ index without using cache"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=device)
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks, model

def query_faq_index(query: str, index, model, chunks, k: int = 3):
    query_vec = model.encode([query], show_progress_bar=False)
    D, I = index.search(query_vec, k)
    similarities = model.similarity(query_vec, model.encode([chunks[i] for i in I[0]], show_progress_bar=False))
    results = [(chunks[i], float(similarities[0][j])) for j, i in enumerate(I[0])]
    return results
