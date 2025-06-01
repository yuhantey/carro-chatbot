import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from google import genai
from google.genai import types
from faq_parser import extract_faq_text
from faq_index import build_faq_index, query_faq_index
import os
import numpy as np
from typing import Literal, Optional
import json

try:
    from gemini_function_schema import search_and_scrape_carro_from_api, extract_function_calls, carro_insight_tool
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

def init_faq_system(pdf_path: str, chunk_size: int = 500):
    """Load and process FAQ document, return index + chunks + model"""
    try:
        st.info("Loading FAQ data...")
        faq_text = extract_faq_text(pdf_path)
        chunks = [faq_text[i:i + chunk_size] for i in range(0, len(faq_text), chunk_size)]
        index, embeddings, chunk_texts, openai_client = build_faq_index(chunks)
        st.success("FAQ data loaded successfully!")
        return index, chunk_texts, openai_client
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

def handle_function_calls(response):
    """Handle function calls from Gemini response with robust error handling"""
    function_results = []
    
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                for i, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        function_call = part.function_call
                        
                        # Safely get function name
                        function_name = getattr(function_call, 'name', None)
                        print(f"DEBUG: Processing function call: {function_name}")
                        
                        if function_name == "search_carro_info":
                            try:
                                args = {}
                                if hasattr(function_call, 'args') and function_call.args:
                                    args = dict(function_call.args)
                                
                                query = args.get("query", "")
                                print(f"DEBUG: Calling search with query='{query}'")
                                
                                search_result = None
                                error_occurred = False
                                
                                if EXTERNAL_API_AVAILABLE:
                                    try:
                                        import inspect
                                        sig = inspect.signature(search_and_scrape_carro_from_api)
                                        param_count = len(sig.parameters)
                                        
                                        if param_count == 1:
                                            # Function only takes query
                                            search_result = search_and_scrape_carro_from_api(query)
                                        else:
                                            # Function takes both query and search_type (fallback)
                                            search_result = search_and_scrape_carro_from_api(query, "general")
                                        
                                        print(f"DEBUG: External API result: {search_result}")
                                        
                                    except Exception as api_error:
                                        print(f"DEBUG: External API error: {api_error}")
                                        search_result = f"Search error: {str(api_error)}"
                                        error_occurred = True
                                else:
                                    search_result = "External search functionality is not available"
                                    error_occurred = True
                                
                                function_results.append({
                                    "name": function_name,
                                    "result": search_result,
                                    "query": query,
                                    "error": error_occurred
                                })
                                
                                print(f"DEBUG: Added function result to list. Total results: {len(function_results)}")
                                
                            except Exception as e:
                                print(f"DEBUG: Error in function execution: {str(e)}")
                                function_results.append({
                                    "name": function_name or "unknown_function",
                                    "result": f"Search function error: {str(e)}",
                                    "query": args.get("query", ""),
                                    "error": True
                                })
                        else:
                            print(f"DEBUG: Unknown function name: {function_name}")
            
    except Exception as e:
        print(f"DEBUG: Error in handle_function_calls: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"DEBUG: Returning {len(function_results)} function results")
    return function_results

def get_answer_from_faq(query: str, index, openai_client, chunk_texts, k: int = 3):
    """Retrieve relevant chunks and generate a response using LLM with function calling"""
    
    results = query_faq_index(query, index, openai_client, chunk_texts, k=k)
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

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt,
            config=config
        )
        
        print(f"DEBUG: Function calling response received")
        
        function_results = handle_function_calls(response)
        print(f"DEBUG: Function results processed: {len(function_results)} calls made")
        
        if function_results and len(function_results) > 0:
            print("DEBUG: Processing function call results...")
            
            # Create follow-up prompt with function results
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

            print("DEBUG: Generating final response with search results...")
            final_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=follow_up_prompt
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
        import traceback
        traceback.print_exc()
        
        try:
            print("DEBUG: Attempting fallback response...")
            fallback_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=f"Based on this FAQ content, please answer the question: {context}\n\nQuestion: {query}"
            )
            
            return ChatResponse(
                message=fallback_response.text.strip(),
                confidence=float(np.mean(similarities)) if similarities else 0.3,
                timestamp=datetime.now(),
                source="FAQ",
                search_used=False
            )
            
        except Exception as fallback_error:
            print(f"DEBUG: Fallback also failed: {str(fallback_error)}")
            
            return ChatResponse(
                message=f"I apologize, but I'm having technical difficulties. The error was: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now(),
                source="Error",
                function_calls_made=None
            )

def render_ui(index, chunk_texts, openai_client):
    """Enhanced Streamlit UI"""
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
                            st.write(f"  Result: {str(call['result'])[:200]}..." if len(str(call['result'])) > 200 else f"  Result: {call['result']}")
    
    if prompt := st.chat_input("Hi! What would you like to know about Carro?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="carro_logo.png"):
            with st.spinner("Thinking..."):
                response = get_answer_from_faq(prompt, index, openai_client, chunk_texts)
            
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
                        st.write(f"  Result: {call['result'][:200]}..." if len(str(call['result'])) > 200 else f"  Result: {call['result']}")
        
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
    """Main application function"""
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
        st.write("• General car buying advice")
        st.write("• Market information")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    pdf_path = "/Users/yuhantey/Downloads/carro-chatbot/faq/carro_malaysia_terms_of_use.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"FAQ file not found: {pdf_path}")
        st.write("Please ensure the FAQ PDF file exists at the specified path.")
        return
    
    index, chunk_texts, openai_client = init_faq_system(pdf_path)
    
    if index is not None and chunk_texts is not None and openai_client is not None:
        render_ui(index, chunk_texts, openai_client)
    else:
        st.error("Failed to initialize the FAQ system. Please check your configuration.")

if __name__ == "__main__":
    main()
