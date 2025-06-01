# Carro Chatbot

An intelligent chatbot for Carro, a online used car dealership, that provides instant answers to customer queries based on FAQ documents on the website. The chatbot respond to questions about used cars, pricing, financing, and other relevant information using the provided FAQ document and internet search APIs for up-to-date data.

## Features

- PDF FAQ document processing
- Semantic search using sentence embeddings
- Integration with Google's Gemini AI
- User-friendly Streamlit interface
- Real-time question answering

## Bonus Features
- Support Multilanguages
- Chatbot UI

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

## Requirements

- Python 3.9
- Streamlit
- Google Generative AI
- Sentence Transformers
- FAISS
- PyPDF2 
- langfuse


---
tags:
- transformers
- xlm-roberta
library_name: transformers
license: cc-by-nc-4.0
language:
  - multilingual
  - af
  - am
  - ar
  - as
  - az
  - be
  - bg
  - bn
  - br
  - bs
  - ca
  - cs
  - cy
  - da
  - de
  - el
  - en
  - eo
  - es
  - et
  - eu
  - fa
  - fi
  - fr
  - fy
  - ga
  - gd
  - gl
  - gu
  - ha
  - he
  - hi
  - hr
  - hu
  - hy
  - id
  - is
  - it
  - ja
  - jv
  - ka
  - kk
  - km
  - kn
  - ko
  - ku
  - ky
  - la
  - lo
  - lt
  - lv
  - mg
  - mk
  - ml
  - mn
  - mr
  - ms
  - my
  - ne
  - nl
  - 'no'
  - om
  - or
  - pa
  - pl
  - ps
  - pt
  - ro
  - ru
  - sa
  - sd
  - si
  - sk
  - sl
  - so
  - sq
  - sr
  - su
  - sv
  - sw
  - ta
  - te
  - th
  - tl
  - tr
  - ug
  - uk
  - ur
  - uz
  - vi
  - xh
  - yi
  - zh
---
Core implementation of Jina XLM-RoBERTa

This implementation is adapted from [XLM-Roberta](https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta). In contrast to the original implementation, this model uses Rotary positional encodings and supports flash-attention 2.

### Models that use this implementation

- [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2)

### Converting weights

Weights from an [original XLMRoberta model](https://huggingface.co/FacebookAI/xlm-roberta-large) can be converted using the `convert_roberta_weights_to_flash.py` script in the model repository.
