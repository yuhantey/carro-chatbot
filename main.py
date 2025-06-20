import streamlit as st
import asyncio
from datetime import datetime
from dataclasses import dataclass
from google import genai
from google.genai import types
from faq.faq_parser import extract_faq_text
from faq.faq_index import build_faq_index, query_faq_index
import os
import numpy as np
from typing import Literal, Optional
import json
import inspect
import traceback

try:
    from utils.external_search.gemini_function_schema import search_and_scrape_carro_from_api, extract_function_calls, carro_insight_tool
    EXTERNAL_API_AVAILABLE = True
    print("DEBUG: External API functions imported successfully")
except ImportError as e:
    print(f"DEBUG: Could not import external API functions: {e}")
    EXTERNAL_API_AVAILABLE = False

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]  # Store this in .streamlit/secrets.toml
client = genai.Client(api_key=GEMINI_API_KEY)

SourceType = Literal["FAQ", "External Search", "Mixed", "Error"]

@dataclass
class ChatResponse:
    """Structure for chatbot response"""
    message: str
    confidence: float
    timestamp: datetime
    source: SourceType
    search_used: bool = False
    function_calls_made: Optional[list] = None

async def init_faq_system_async(pdf_path: str, chunk_size: int = 500):
    """Load and process FAQ document asynchronously, return index + chunks + model"""
    try:
        st.info("Loading FAQ data...")
        
        # Run CPU-bound operations in thread pool
        loop = asyncio.get_event_loop()
        faq_text = await loop.run_in_executor(None, extract_faq_text, pdf_path)
        
        chunks = [faq_text[i:i + chunk_size] for i in range(0, len(faq_text), chunk_size)]
        
        # Build FAQ index asynchronously
        index, embeddings, chunk_texts, gemini_client = await loop.run_in_executor(
            None, build_faq_index, chunks
        )
        
        st.success("FAQ data loaded successfully!")
        return index, chunk_texts, gemini_client
    except Exception as e:
        st.error(f"Failed to load FAQ data: {str(e)}")
        return None, None, None

def create_valid_function_tool():
    """Create a properly formatted function tool for Gemini API"""
    function_declaration = types.FunctionDeclaration(
        name="search_carro_info",
        description="Search for current Carro car listings, prices, market information, and opening hours",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for car information, opening hours, or other Carro-related information"
                }
            },
            "required": ["query"]
        }
    )
    
    return types.Tool(function_declarations=[function_declaration])

async def execute_search_function_async(query: str):
    """Execute search function asynchronously"""
    try:
        if not EXTERNAL_API_AVAILABLE:
            return "External search functionality is not available", True
        
        if asyncio.iscoroutinefunction(search_and_scrape_carro_from_api):
            search_result = await search_and_scrape_carro_from_api(query)
        else:
            loop = asyncio.get_event_loop()
            sig = inspect.signature(search_and_scrape_carro_from_api)
            param_count = len(sig.parameters)
            
            if param_count == 1:
                search_result = await loop.run_in_executor(
                    None, search_and_scrape_carro_from_api, query
                )
            else:
                search_result = await loop.run_in_executor(
                    None, search_and_scrape_carro_from_api, query, "general"
                )
        
        print(f"DEBUG: External API result: {search_result}")
        return search_result, False
        
    except Exception as api_error:
        print(f"DEBUG: External API error: {api_error}")
        return f"Search error: {str(api_error)}", True

async def handle_function_calls_async(response):
    """Handle function calls from Gemini response asynchronously with robust error handling"""
    function_results = []
    
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                function_calls_to_execute = []
                
                for i, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        function_call = part.function_call
                        function_name = getattr(function_call, 'name', None)
                        
                        if function_name == "search_carro_info":
                            args = {}
                            if hasattr(function_call, 'args') and function_call.args:
                                args = dict(function_call.args)
                            
                            query = args.get("query", "")
                            function_calls_to_execute.append((function_name, query))
                
                if function_calls_to_execute:
                    tasks = []
                    for function_name, query in function_calls_to_execute:
                        print(f"DEBUG: Preparing async call for query='{query}'")
                        task = execute_search_function_async(query)
                        tasks.append((function_name, query, task))
                    
                    for function_name, query, task in tasks:
                        try:
                            search_result, error_occurred = await task
                            
                            function_results.append({
                                "name": function_name,
                                "result": search_result,
                                "query": query,
                                "error": error_occurred
                            })
                            
                            print(f"DEBUG: Added async function result. Total results: {len(function_results)}")
                            
                        except Exception as e:
                            print(f"DEBUG: Error in async function execution: {str(e)}")
                            function_results.append({
                                "name": function_name,
                                "result": f"Search function error: {str(e)}",
                                "query": query,
                                "error": True
                            })
            
    except Exception as e:
        print(f"DEBUG: Error in handle_function_calls_async: {str(e)}")
        traceback.print_exc()
    
    print(f"DEBUG: Returning {len(function_results)} async function results")
    return function_results

async def generate_content_async(model: str, contents: str, config=None):
    """Generate content asynchronously"""
    loop = asyncio.get_event_loop()
    
    if config:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        )
    else:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=contents
            )
        )
    
    return response

async def get_answer_from_faq_async(query: str, index, gemini_client, chunk_texts, k: int = 3):
    """Retrieve relevant chunks and generate a response using LLM with async function calling"""
    
    # Run FAQ query in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, query_faq_index, query, index, gemini_client, chunk_texts, k
    )
    
    chunks = [chunk for chunk, _ in results]
    similarities = [score for _, score in results]
    context = "\n\n".join(chunks)
    
    prompt = f"""You are a helpful assistant for Carro, a used car company in Malaysia. 

    You have access to FAQ content and can search for current car information when needed.

    FAQ Context:
    {context}

    Guidelines:
    - Answer clearly and concisely based on the FAQ content when possible
    - If the question requires current market data, car listings, pricing information, opening hours, general car buying advice, market information or contact information that's not in the FAQ, use the search function
    - If the question is beyond the scope of carro FAQ and not related to carro, politely indicate you don't know
    - Always be helpful and professional

    Question: {query}

    Please provide a comprehensive answer."""

    try:
        tools = [create_valid_function_tool()]
        print(f"DEBUG: Created function tool for query: {query}")
        
        config = types.GenerateContentConfig(
            tools=tools,
            temperature=0.7,
            max_output_tokens=1024
        )

        response = await generate_content_async(
            "gemini-2.5-flash-preview-04-17",
            prompt,
            config
        )
        
        print(f"DEBUG: Async function calling response received")
        
        function_results = await handle_function_calls_async(response)
        print(f"DEBUG: Async function results processed: {len(function_results)} calls made")
        
        if function_results and len(function_results) > 0:
            print("DEBUG: Processing async function call results...")
            
            function_context = "\n".join([
                f"Search for '{result['query']}' returned: {result['result']}" 
                for result in function_results
            ])
            
            follow_up_prompt = f"""Based on the FAQ context and the search results below, provide a comprehensive answer to the user's question.

            FAQ Context:
            {context}

            Search Results:
            {function_context}

            Original Question: {query}

            Please synthesize the information from both the FAQ and search results to provide a helpful, accurate response. If the search results contain relevant information, prioritize that for current/dynamic information like opening hours, contact details, or current listings."""

            print("DEBUG: Generating final async response with search results...")
            final_response = await generate_content_async(
                "gemini-2.5-flash-preview-04-17",
                follow_up_prompt
            )
            
            return ChatResponse(
                message=final_response.text.strip() if final_response.text else "No response generated",
                confidence=float(np.mean(similarities)) if similarities else 0.7,
                timestamp=datetime.now(),
                source="Mixed",
                search_used=True,
                function_calls_made=function_results
            )
        
        else:
            print("DEBUG: No function calls made, using FAQ response")
            response_text = response.text if hasattr(response, 'text') and response.text else "I couldn't generate a response based on the available information."
            
            return ChatResponse(
                message=response_text.strip(),
                confidence=float(np.mean(similarities)) if similarities else 0.5,
                timestamp=datetime.now(),
                source="FAQ",
                search_used=False
            )
            
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        traceback.print_exc()
        
        try:
            print("DEBUG: Attempting async fallback response...")
            fallback_response = await generate_content_async(
                "gemini-2.5-flash-preview-04-17",
                f"Based on this FAQ content, please answer the question: {context}\n\nQuestion: {query}"
            )
            
            return ChatResponse(
                message=fallback_response.text.strip(),
                confidence=float(np.mean(similarities)) if similarities else 0.3,
                timestamp=datetime.now(),
                source="FAQ",
                search_used=False
            )
            
        except Exception as fallback_error:
            print(f"DEBUG: Async fallback also failed: {str(fallback_error)}")
            
            return ChatResponse(
                message=f"I apologize, but I'm having technical difficulties. The error was: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                source="Error",
                function_calls_made=None
            )

def run_async(coro):
    """Helper function to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)  # 30 second timeout
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)

async def render_ui_async(index, chunk_texts, gemini_client):
    """Enhanced Streamlit UI with async support"""
    st.title("🚗 Carro Malaysia Chatbot")
    st.markdown("*Ask me anything about used cars, pricing, policies, or current market information!*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("Response Details"):
                    metadata = message["metadata"]
                    st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                    st.write(f"**Confidence:** {metadata.get('confidence', 0):.2f}")
                    st.write(f"**Search Used:** {metadata.get('search_used', False)}")
                    if metadata.get('function_calls_made'):
                        st.write("**Function Calls Made:**")
                        for call in metadata['function_calls_made']:
                            st.write(f"- {call['name']}: {call['query']}")
                            result_text = str(call['result'])
                            display_result = f"{result_text[:200]}..." if len(result_text) > 200 else result_text
                            st.write(f"  Result: {display_result}")
    
    if prompt := st.chat_input("Hi! What would you like to know about Carro?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="utils/images/carro_logo.png"):
            with st.spinner("Thinking..."):
                response = await get_answer_from_faq_async(prompt, index, gemini_client, chunk_texts)
            
            st.markdown(response.message)
            
            with st.expander("Response Details"):
                st.write(f"**Source:** {response.source}")
                st.write(f"**Confidence:** {response.confidence:.2f}")
                st.write(f"**Search Used:** {response.search_used}")
                st.write(f"**Timestamp:** {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if response.function_calls_made:
                    st.write("**External Searches Performed:**")
                    for call in response.function_calls_made:
                        st.write(f"- Function: {call['name']}")
                        st.write(f"  Query: {call['query']}")
                        result_text = str(call['result'])
                        display_result = f"{result_text[:200]}..." if len(result_text) > 200 else result_text
                        st.write(f"  Result: {display_result}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.message,
            "metadata": {
                "source": response.source,
                "confidence": response.confidence,
                "search_used": response.search_used,
                "function_calls_made": response.function_calls_made
            }
        })

def main():
    """Main application function with async support"""
    st.set_page_config(
        page_title="Carro Chatbot",
        page_icon="🚗",
        layout="wide"
    )
    
    with st.sidebar:
        st.header("About Carro Chatbot")
        st.write("This chatbot can help you with:")
        st.write("• FAQ about Carro policies")
        st.write("• Current car listings and prices")
        st.write("• Opening hours and contact information")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    pdf_path = "/Users/yuhantey/Downloads/carro-chatbot/faq/carro_malaysia_terms_of_use.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"FAQ file not found: {pdf_path}")
        st.write("Please ensure the FAQ PDF file exists at the specified path.")
        return
    
    if "faq_initialized" not in st.session_state:
        with st.spinner("Initializing FAQ system..."):
            index, chunk_texts, gemini_client = run_async(init_faq_system_async(pdf_path))
            
            if index is not None and chunk_texts is not None and gemini_client is not None:
                st.session_state.faq_initialized = True
                st.session_state.index = index
                st.session_state.chunk_texts = chunk_texts
                st.session_state.gemini_client = gemini_client
            else:
                st.error("Failed to initialize the FAQ system. Please check your configuration.")
                return
    
    if st.session_state.get("faq_initialized", False):
        run_async(render_ui_async(
            st.session_state.index,
            st.session_state.chunk_texts,
            st.session_state.gemini_client
        ))

if __name__ == "__main__":
    main()