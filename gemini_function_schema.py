import requests
import streamlit as st
from google.genai import types
from typing import Any
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import re

GOOGLE_SEARCH_KEY = st.secrets["GOOGLE_SEARCH_ENGINE"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

get_carro_realtime = types.FunctionDeclaration(
    name="search_and_scrape_carro",
    description="Search for Malaysia car-related information and scrape detailed content from relevant websites",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="The search query for car-related information in Malaysia"
            ),
            "scrape_content": types.Schema(
                type=types.Type.BOOLEAN,
                description="Whether to scrape full content from found websites (default: True)"
            )
        },
        required=["query"]
    )
)

# Tool wrapper
carro_insight_tool = types.Tool(
    function_declarations=[get_carro_realtime]
)

def get_page_content(url, timeout=10):
    """Scrape content from a single webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid token limits
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Error scraping {url}: {str(e)}"
    except Exception as e:
        return f"Error processing {url}: {str(e)}"

def search_and_scrape_carro_from_api(query, scrape_content=True):
    """Enhanced function that searches and optionally scrapes content"""
    api_key = GOOGLE_SEARCH_KEY
    cse_id = GOOGLE_CSE_ID
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 5
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        print(f'DEBUG ==== Google search results for: {query}')

        if "items" not in results or len(results["items"]) == 0:
            return "Sorry, I couldn't find any relevant information."
        
        formatted_results = f"Search Results for '{query}':\n{'='*50}\n\n"
        
        for i, item in enumerate(results["items"], 1):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description")
            
            formatted_results += f"🔍 **Result {i}: {title}**\n"
            formatted_results += f"🔗 URL: {link}\n"
            formatted_results += f"📝 Preview: {snippet}\n"
            
            if scrape_content and link != "No link":
                print(f"Scraping content from: {link}")
                scraped_content = get_page_content(link)
                
                if scraped_content and not scraped_content.startswith("Error"):
                    formatted_results += f"📄 **Full Content:**\n{scraped_content}\n"
                else:
                    formatted_results += f"⚠️ Could not scrape content: {scraped_content}\n"
                
                # Add delay to be respectful
                time.sleep(1)
            
            formatted_results += f"{'-'*50}\n\n"
        
        formatted_results += f"**Summary:**\n"
        formatted_results += f"Found {len(results['items'])} results for your query about '{query}'\n"
        if scrape_content:
            formatted_results += f"Full content has been scraped from available websites.\n"
        
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        print(f"Search API error: {str(e)}")
        return f"Search failed due to API error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

def search_specific_website(query, target_domain=None):
    """Search and scrape from a specific website"""
    if target_domain:
        search_query = f"site:{target_domain} {query}"
    else:
        search_query = query
    
    return search_and_scrape_carro_from_api(search_query, scrape_content=True)

def extract_contact_info(scraped_content):
    """Extract contact information from scraped content"""
    contact_patterns = {
        'phone': r'(\+?6?0?1[0-9-\s]{8,12})',
        'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        'address': r'([A-Z][a-zA-Z\s,0-9-]+(?:Malaysia|Kuala Lumpur|Selangor|Johor))',
    }
    
    extracted_info = {}
    
    for info_type, pattern in contact_patterns.items():
        matches = re.findall(pattern, scraped_content)
        if matches:
            extracted_info[info_type] = list(set(matches))
    
    return extracted_info

function_handler = {
    "search_and_scrape_carro": search_and_scrape_carro_from_api
}

def extract_function_calls(response) -> list[dict]:
    """Extract function calls from Gemini response"""
    function_calls: list[dict] = []
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        
        if hasattr(candidate, 'function_calls') and candidate.function_calls:
            for function_call in candidate.function_calls:
                function_call_dict: dict[str, dict[str, Any]] = {function_call.name: {}}
                for key, value in function_call.args.items():
                    function_call_dict[function_call.name][key] = value
                function_calls.append(function_call_dict)
        
        elif hasattr(candidate, 'content') and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call_dict = {part.function_call.name: dict(part.function_call.args)}
                    function_calls.append(function_call_dict)
    
    return function_calls

def execute_function_call(function_call):
    """Execute a function call and return the result"""
    function_name = function_call.name
    function_args = dict(function_call.args)
    
    if function_name in function_handler:
        return function_handler[function_name](**function_args)
    else:
        return f"Unknown function: {function_name}"

# # Test functions
# def test_search_and_scrape(query="Carro Malaysia contact information"):
#     """Test the enhanced search and scrape functionality"""
#     print(f"Testing enhanced search and scrape for: {query}")
#     result = search_and_scrape_carro_from_api(query, scrape_content=True)
#     print("=== ENHANCED RESULT ===")
#     print(result)
#     print("=======================")
#     return result

# def test_specific_site_scrape(query="contact us", domain="carro.my"):
#     """Test scraping from a specific website"""
#     print(f"Testing specific site scrape: {query} from {domain}")
#     result = search_specific_website(query, domain)
#     print("=== SPECIFIC SITE RESULT ===")
#     print(result)
#     print("============================")
#     return result

# # Usage examples:
# if __name__ == "__main__":
#     # Test the enhanced functionality
#     test_search_and_scrape("Carro Malaysia office hours")
#     test_specific_site_scrape("contact information", "carro.my")