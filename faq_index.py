# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import torch
# import os
# import pickle

# MODEL_CACHE_DIR = "model_cache"
# EMBEDDINGS_FILE = os.path.join(MODEL_CACHE_DIR, "embeddings.pkl")
# INDEX_FILE = os.path.join(MODEL_CACHE_DIR, "faiss_index.bin")
# CHUNKS_FILE = os.path.join(MODEL_CACHE_DIR, "chunks.pkl")

# @st.cache_resource
# def build_faq_index(chunks: list[str]):
#     """Build FAQ index without using cache"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=device)
#     embeddings = model.encode(chunks, show_progress_bar=True)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))
#     return index, embeddings, chunks, model

# def query_faq_index(query: str, index, model, chunks, k: int = 3):
#     query_vec = model.encode([query], show_progress_bar=False)
#     D, I = index.search(query_vec, k)
#     similarities = model.similarity(query_vec, model.encode([chunks[i] for i in I[0]], show_progress_bar=False))
#     results = [(chunks[i], float(similarities[0][j])) for j, i in enumerate(I[0])]
#     return results


import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import threading

MODEL_CACHE_DIR = "model_cache"
EMBEDDINGS_FILE = os.path.join(MODEL_CACHE_DIR, "embeddings.pkl")
INDEX_FILE = os.path.join(MODEL_CACHE_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(MODEL_CACHE_DIR, "chunks.pkl")

# Thread-local storage for models to avoid thread safety issues
_thread_local = threading.local()

def _get_model():
    """Get model instance for current thread"""
    if not hasattr(_thread_local, 'model'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _thread_local.model = SentenceTransformer(
            "jinaai/jina-embeddings-v3", 
            trust_remote_code=True, 
            device=device
        )
    return _thread_local.model

def _encode_chunks_sync(chunks: List[str]) -> np.ndarray:
    """Synchronous function to encode chunks"""
    model = _get_model()
    return model.encode(chunks, show_progress_bar=True)

def _encode_query_sync(query: str) -> np.ndarray:
    """Synchronous function to encode single query"""
    model = _get_model()
    return model.encode([query], show_progress_bar=False)

def _build_faiss_index_sync(embeddings: np.ndarray):
    """Synchronous function to build FAISS index"""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def _search_index_sync(index, query_vec: np.ndarray, k: int):
    """Synchronous function to search FAISS index"""
    D, I = index.search(query_vec, k)
    return D, I

def _calculate_similarities_sync(query_vec: np.ndarray, chunk_embeddings: np.ndarray):
    """Synchronous function to calculate similarities"""
    model = _get_model()
    return model.similarity(query_vec, chunk_embeddings)

@st.cache_resource
async def build_faq_index_async(chunks: List[str]):
    """Build FAQ index asynchronously"""
    try:
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # Check if cached files exist
        if (os.path.exists(EMBEDDINGS_FILE) and 
            os.path.exists(INDEX_FILE) and 
            os.path.exists(CHUNKS_FILE)):
            
            try:
                # Load from cache asynchronously
                loop = asyncio.get_event_loop()
                
                # Load cached data in parallel
                load_tasks = [
                    loop.run_in_executor(None, _load_embeddings),
                    loop.run_in_executor(None, _load_index),
                    loop.run_in_executor(None, _load_chunks)
                ]
                
                embeddings, index, cached_chunks = await asyncio.gather(*load_tasks)
                
                # Verify chunks match
                if cached_chunks == chunks:
                    print("Loading FAQ index from cache...")
                    # Get model instance
                    model = await loop.run_in_executor(None, _get_model)
                    return index, embeddings, chunks, model
                else:
                    print("Chunks changed, rebuilding index...")
            except Exception as e:
                print(f"Cache loading failed: {e}, rebuilding...")
        
        print("Building new FAQ index...")
        
        # Run embedding generation in thread pool
        loop = asyncio.get_event_loop()
        
        with st.spinner("Generating embeddings..."):
            embeddings = await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1), 
                _encode_chunks_sync, 
                chunks
            )
        
        with st.spinner("Building FAISS index..."):
            index = await loop.run_in_executor(
                None, 
                _build_faiss_index_sync, 
                embeddings
            )
        
        # Get model instance
        model = await loop.run_in_executor(None, _get_model)
        
        # Save to cache asynchronously
        cache_tasks = [
            loop.run_in_executor(None, _save_embeddings, embeddings),
            loop.run_in_executor(None, _save_index, index),
            loop.run_in_executor(None, _save_chunks, chunks)
        ]
        
        # Don't wait for caching to complete - run in background
        asyncio.create_task(_save_cache_async(cache_tasks))
        
        return index, embeddings, chunks, model
        
    except Exception as e:
        print(f"Error building FAQ index: {e}")
        raise

def _load_embeddings():
    """Load embeddings from cache"""
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def _load_index():
    """Load FAISS index from cache"""
    return faiss.read_index(INDEX_FILE)

def _load_chunks():
    """Load chunks from cache"""
    with open(CHUNKS_FILE, 'rb') as f:
        return pickle.load(f)

def _save_embeddings(embeddings):
    """Save embeddings to cache"""
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

def _save_index(index):
    """Save FAISS index to cache"""
    faiss.write_index(index, INDEX_FILE)

def _save_chunks(chunks):
    """Save chunks to cache"""
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump(chunks, f)

async def _save_cache_async(cache_tasks):
    """Save cache files asynchronously in background"""
    try:
        await asyncio.gather(*cache_tasks)
        print("Cache saved successfully")
    except Exception as e:
        print(f"Cache saving failed: {e}")

async def query_faq_index_async(query: str, index, model, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Query FAQ index asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        
        # Encode query asynchronously
        query_vec = await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1),
            _encode_query_sync,
            query
        )
        
        # Search index asynchronously
        D, I = await loop.run_in_executor(
            None,
            _search_index_sync,
            index,
            query_vec,
            k
        )
        
        # Get relevant chunks
        relevant_chunks = [chunks[i] for i in I[0]]
        
        # Encode relevant chunks for similarity calculation
        chunk_embeddings = await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1),
            _encode_chunks_sync,
            relevant_chunks
        )
        
        # Calculate similarities asynchronously
        similarities = await loop.run_in_executor(
            None,
            _calculate_similarities_sync,
            query_vec,
            chunk_embeddings
        )
        
        # Prepare results
        results = [
            (chunks[i], float(similarities[0][j])) 
            for j, i in enumerate(I[0])
        ]
        
        return results
        
    except Exception as e:
        print(f"Error querying FAQ index: {e}")
        # Fallback to synchronous operation
        return query_faq_index_sync(query, index, model, chunks, k)

def query_faq_index_sync(query: str, index, model, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Synchronous fallback for FAQ querying"""
    try:
        query_vec = model.encode([query], show_progress_bar=False)
        D, I = index.search(query_vec, k)
        similarities = model.similarity(
            query_vec, 
            model.encode([chunks[i] for i in I[0]], show_progress_bar=False)
        )
        results = [(chunks[i], float(similarities[0][j])) for j, i in enumerate(I[0])]
        return results
    except Exception as e:
        print(f"Error in sync query: {e}")
        return []

# Wrapper functions for backward compatibility
@st.cache_resource
def build_faq_index(chunks: List[str]):
    """Synchronous wrapper for backward compatibility"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, build_faq_index_async(chunks))
                return future.result(timeout=300)  # 5 minute timeout
        else:
            return loop.run_until_complete(build_faq_index_async(chunks))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(build_faq_index_async(chunks))
    except Exception as e:
        print(f"Async build failed, falling back to sync: {e}")
        # Fallback to original sync implementation
        return build_faq_index_sync(chunks)

def build_faq_index_sync(chunks: List[str]):
    """Original synchronous implementation as fallback"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=device)
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks, model

def query_faq_index(query: str, index, model, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Synchronous wrapper for backward compatibility"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    query_faq_index_async(query, index, model, chunks, k)
                )
                return future.result(timeout=30)  # 30 second timeout
        else:
            return loop.run_until_complete(query_faq_index_async(query, index, model, chunks, k))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(query_faq_index_async(query, index, model, chunks, k))
    except Exception as e:
        print(f"Async query failed, falling back to sync: {e}")
        # Fallback to sync implementation
        return query_faq_index_sync(query, index, model, chunks, k)

# Utility function for running async operations
async def run_faq_operations_async(chunks: List[str], queries: List[str], k: int = 3):
    """Run multiple FAQ operations concurrently"""
    # Build index first
    index, embeddings, chunks, model = await build_faq_index_async(chunks)
    
    # Run multiple queries concurrently
    query_tasks = [
        query_faq_index_async(query, index, model, chunks, k)
        for query in queries
    ]
    
    results = await asyncio.gather(*query_tasks, return_exceptions=True)
    
    return index, model, results