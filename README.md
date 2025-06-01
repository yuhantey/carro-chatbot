# Carro Chatbot

An intelligent chatbot for Carro, a online used car dealership, that provides instant answers to customer queries based on FAQ documents on the website. The chatbot respond to questions about used cars, pricing, financing, and other relevant information using the provided FAQ document and internet search APIs for up-to-date data.

## Features

- PDF FAQ document processing
- Semantic search using sentence embeddings
- Integration with Google's Gemini AI
- Real-time question answering

## Bonus Features
- Support Multilanguages
- User-friendly Streamlit Chatbot UI interface
- Websrapping for Static sites
- UI Dashboard for monitoring LLM input, output, Scores, Model costs

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yuhantey/carro-chatbot.git
cd carro-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv carro
source carro/bin/activate  # On Windows: carro\Scripts\activate
```

or uv
```bash
uv venv carro
source carro/bin/activate  # On Windows: carro\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your Google API key:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

5. Run the application:
```bash
streamlit run main.py
```

## Project Structure

- `main.py`: Main application and Streamlit UI
- `faq_parser.py`: PDF text extraction
- `faq_index.py`: Vector embeddings and similarity search
- `requirements.txt`: Project dependencies
- `.streamlit/secrets.toml`: Configuration and API keys


## Testing & Evaluation

1. Clone Langfuse Repository
```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

2. Start the application

```bash
docker compose up
```

3. Done! Open http://localhost:3000 in your browser to access the Langfuse UI.

4. Generate the Secret key & Public key to pass in test_carro_chatbot_langfuse.py.

## Requirements

- Python 3.9
- Streamlit
- Google Generative AI
- Sentence Transformers
- FAISS
- PyPDF2 
- langfuse