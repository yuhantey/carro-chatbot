import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from google import genai
from faq_parser import extract_faq_text
from faq_index import build_faq_index, query_faq_index
import os
import numpy as np

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]  # Store this in .streamlit/secrets.toml
client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class ChatResponse:
    """Structure for chatbot response"""
    message: str
    confidence: float
    timestamp: datetime
    source: str = "FAQ"
    search_used: bool = False

def init_faq_system(pdf_path: str, chunk_size: int = 500):
    """Load and process FAQ document, return index + chunks + model"""
    st.info("Loading FAQ data...")
    faq_text = extract_faq_text(pdf_path)
    chunks = [faq_text[i:i + chunk_size] for i in range(0, len(faq_text), chunk_size)]
    index, embeddings, chunk_texts, openai_client = build_faq_index(chunks)
    return index, chunk_texts, openai_client

def get_answer_from_faq(query: str, index, openai_client, chunk_texts, k: int = 3):
    """Retrieve relevant chunks and generate a response using LLM"""
    results = query_faq_index(query, index, openai_client, chunk_texts, k=k)
    
    chunks = [chunk for chunk, _ in results]
    similarities = [score for _, score in results]
    
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant for Carro, a used car company. 
    Answer clearly and concisely based on the following FAQ content:
    Searching for current information using external APIs when necessary.
    Communicating clearly and politely with users.
    If the quetion is beyong the scope of the document, you should response politely indicating you don't know.
    {context}

    Question: {query}
    Answer:"""

    print(f'DEBUG context {context}')
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return ChatResponse(
            message=response.text.strip(),
            confidence=float(np.mean(similarities)),
            timestamp=datetime.now(),
            source="FAQ"
        )
    except Exception as e:
        return ChatResponse(
            message=f"Sorry, I encountered an issue answering your question: {str(e)}",
            confidence=0.0,
            timestamp=datetime.now(),
            source="error"
        )

def render_ui(index, chunk_texts, openai_client):
    """Streamlit UI"""
    st.title("Carro Chatbot 🚗")
    query = st.text_input("Ask me anything about used cars, pricing, or policies:")

    if query:
        response = get_answer_from_faq(query, index, openai_client, chunk_texts)
        st.markdown(f"**Answer:** {response.message}")
        st.caption(f"Source: {response.source} | Confidence: {response.confidence:.1f}")

def main():
    index, chunk_texts, openai_client = init_faq_system("/Users/yuhantey/Downloads/carro-chatbot/faq/carro_malaysia_terms_of_use.pdf")
    render_ui(index, chunk_texts, openai_client)

if __name__ == "__main__":
    main()