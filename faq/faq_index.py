import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import pickle

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
