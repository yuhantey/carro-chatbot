# Carro Chatbot Architecture

## Overview
The Carro Chatbot is a sophisticated conversational AI system designed to provide information about Carro's services, car listings, and general automotive information in Malaysia. The system combines FAQ-based knowledge with real-time web search capabilities to deliver accurate and up-to-date responses.

## System Components

### 1. Core Components

#### 1.1 FAQ Processing System
- **File**: `faq_parser.py`
- **Purpose**: Extracts text content from PDF documents
- **Key Features**:
  - Uses PyPDF2 for PDF text extraction
  - Processes documents page by page
  - Maintains text formatting and structure

#### 1.2 FAQ Indexing System
- **File**: `faq_index.py`
- **Purpose**: Creates and manages a searchable index of FAQ content
- **Key Features**:
  - Uses SentenceTransformer for text embeddings
  - Implements FAISS for efficient similarity search
  - Supports semantic search capabilities
  - Caches model artifacts for performance

#### 1.3 External Search System
- **File**: `gemini_function_schema.py`
- **Purpose**: Provides real-time web search and content scraping
- **Key Features**:
  - Google Custom Search API integration
  - Web content scraping with BeautifulSoup
  - Contact information extraction
  - Rate limiting and error handling

#### 1.4 Main Application
- **File**: `main.py`
- **Purpose**: Orchestrates the entire system and provides the user interface
- **Key Features**:
  - Streamlit-based web interface
  - Chat history management
  - Response confidence scoring
  - Hybrid response generation

## Architecture Decisions

### 1. Hybrid Knowledge Base
- **Decision**: Combine static FAQ with dynamic web search
- **Rationale**:
  - FAQ provides reliable, curated information
  - Web search enables access to current information
  - Hybrid approach ensures comprehensive coverage

### 2. Vector Search Implementation
- **Decision**: Use FAISS with SentenceTransformer
- **Rationale**:
  - Efficient similarity search
  - Semantic understanding of queries
  - Scalable for large document sets

### 3. Response Generation Strategy
- **Decision**: Multi-stage response generation
- **Rationale**:
  - First checks FAQ knowledge base
  - Falls back to web search when needed
  - Combines multiple sources for comprehensive answers

### 4. Error Handling and Fallbacks
- **Decision**: Implement multiple fallback mechanisms
- **Rationale**:
  - Ensures system reliability
  - Graceful degradation of service
  - Maintains user experience during failures

## Data Flow

1. **User Query Processing**
   - User input received through Streamlit interface
   - Query preprocessed and normalized

2. **Knowledge Base Search**
   - Query embedded using SentenceTransformer
   - FAISS index searched for relevant FAQ chunks
   - Confidence scores calculated

3. **External Search (if needed)**
   - Google Custom Search API queried
   - Relevant pages scraped and processed
   - Information extracted and formatted

4. **Response Generation**
   - FAQ and search results combined
   - Gemini model generates final response
   - Response formatted and presented to user

## Security Considerations

1. **API Key Management**
   - Keys stored in Streamlit secrets
   - No hardcoded credentials
   - Secure access to external services

2. **Rate Limiting**
   - Implemented for web scraping
   - Respects website terms of service
   - Prevents abuse of external APIs

## Performance Optimizations

1. **Caching**
   - Model artifacts cached
   - Embeddings stored for reuse
   - Reduces computation overhead

2. **Efficient Search**
   - FAISS for fast similarity search
   - Chunked document processing
   - Optimized vector operations

## Future Improvements

1. **Knowledge Base Enhancement**
   - Add more FAQ sources
   - Implement document versioning
   - Support multiple languages

2. **Search Capabilities**
   - Add more specialized search functions
   - Improve content extraction
   - Enhance result ranking

3. **User Experience**
   - Add conversation memory
   - Implement user feedback system
   - Enhance response personalization

## Dependencies

- **Core Libraries**:
  - Streamlit: Web interface
  - SentenceTransformer: Text embeddings
  - FAISS: Vector search
  - PyPDF2: PDF processing
  - BeautifulSoup: Web scraping
  - Google Generative AI: Response generation

## Conclusion
The Carro Chatbot architecture demonstrates a robust approach to building a hybrid knowledge system that combines static FAQ content with dynamic web search capabilities. The system's modular design allows for easy maintenance and future enhancements while maintaining high performance and reliability. 