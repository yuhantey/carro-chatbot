# test_chatbot_simple_langfuse.py
import os
import time
from datetime import datetime
from langfuse import Langfuse
from main import get_answer_from_faq, init_faq_system
import streamlit as st



# local Configuration
# LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE_SECRET_KEY"]
# LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE_PUBLIC_KEY"]
# LANGFUSE_HOST = "http://localhost:3000"

# Cloud Configuration
langfuse_client = Langfuse(
    public_key=st.secrets["LANGFUSE_CLOUD_PUBLIC_KEY"],
    secret_key=st.secrets["LANGFUSE_CLOUD_SECRET_KEY"],
    host="https://us.cloud.langfuse.com"
)

class CarroChatbot:
    def __init__(self):
        pdf_path = "/Users/yuhantey/Downloads/carro-chatbot/faq/carro_malaysia_terms_of_use.pdf"
        self.index, self.chunk_texts, self.openai_client = init_faq_system(pdf_path)
        if not all([self.index, self.chunk_texts, self.openai_client]):
            raise Exception("Failed to initialize FAQ system")

    def get_response(self, question: str) -> str:
        """Wrapper method to match the test interface"""
        response = get_answer_from_faq(question, self.index, self.openai_client, self.chunk_texts)
        return response.message

class ChatbotTesterWithLangfuse:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.langfuse = langfuse_client
        
    def run_simple_tests(self):
        """Run basic tests with Cloud Langfuse tracking"""
        print("🧪 Testing Chatbot with Langfuse Cloud...")
        print(f"☁️  Using Langfuse Cloud (US Region)")
        print("=" * 50 + "\n")
        
        # Create a test session for better organization
        self.test_session_id = f"test-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.test_faq()
        self.test_search()
        self.test_out_of_scope()
        
        # Ensure all data is sent to cloud
        self.langfuse.flush()
        
        print(f"\n✅ Tests complete!")
        print(f"📊 View results at: https://us.cloud.langfuse.com")
        print(f"🔍 Session ID: {self.test_session_id}")
        
    def test_faq(self):
        """Test FAQ question"""
        question = "What services does Carro provide?"
        print(f"📋 Testing FAQ: {question}")
        
        trace = self.langfuse.trace(
            name="test-faq",
            input=question,
            session_id=self.test_session_id,
            metadata={
                "test_type": "FAQ",
                "timestamp": datetime.now().isoformat(),
                "environment": "test"
            },
            tags=["test", "faq"]
        )
        
        generation = trace.generation(
            name="gemini-response",
            model="gemini-1.5-flash",
            input=question
        )
        
        start = time.time()
        try:
            response = self.chatbot.get_response(question)
            response_time = time.time() - start
            
            generation.end(
                output=response,
                usage={
                    "total_tokens": len(response.split()),
                    "response_time_seconds": response_time
                }
            )
            
            # Update trace with output
            trace.update(
                output=response,
                metadata={
                    "response_time": response_time,
                    "response_length": len(response)
                }
            )
            
            # Score the test
            keywords = ["sell", "buy", "vehicle", "service"]
            found = sum(1 for k in keywords if k in response.lower())
            accuracy = found / len(keywords)
            
            trace.score(
                name="accuracy",
                value=accuracy,
                comment=f"Found {found}/{len(keywords)} keywords"
            )
            
            print(f"✅ Response in {response_time:.2f}s - Accuracy: {accuracy*100:.0f}%")
            
        except Exception as e:
            generation.end(
                level="ERROR",
                status_message=str(e)
            )
            print(f"❌ Error: {str(e)}")
        
    def test_search(self):
        """Test search question"""
        question = "What is the current price of Mazda CX-5 in Malaysia?"
        print(f"\n🔍 Testing Search: {question}")
        
        trace = self.langfuse.trace(
            name="test-search",
            input=question,
            session_id=self.test_session_id,
            metadata={"test_type": "search"},
            tags=["test", "search"]
        )
        
        generation = trace.generation(
            name="gemini-search",
            model="gemini-1.5-flash",
            input=question
        )
        
        start = time.time()
        try:
            response = self.chatbot.get_response(question)
            response_time = time.time() - start
            
            generation.end(
                output=response,
                usage={
                    "total_tokens": len(response.split()),
                    "response_time_seconds": response_time
                }
            )
            
            # Update trace with output
            has_price = "RM" in response or "price" in response.lower()
            trace.update(
                output=response,
                metadata={
                    "response_time": response_time,
                    "has_price": has_price
                }
            )
            
            # Check for price info
            trace.score(
                name="has_price",
                value=1.0 if has_price else 0.0
            )
            
            print(f"✅ Response in {response_time:.2f}s - Has price: {has_price}")
            
        except Exception as e:
            generation.end(
                level="ERROR",
                status_message=str(e)
            )
            print(f"❌ Error: {str(e)}")
        
    def test_out_of_scope(self):
        """Test out-of-scope question"""
        question = "What's the weather today?"
        print(f"\n🚫 Testing Out-of-scope: {question}")
        
        trace = self.langfuse.trace(
            name="test-out-of-scope",
            input=question,
            session_id=self.test_session_id,
            metadata={"test_type": "out_of_scope"},
            tags=["test", "out-of-scope"]
        )
        
        generation = trace.generation(
            name="gemini-out-of-scope",
            model="gemini-1.5-flash",
            input=question
        )
        
        start = time.time()
        try:
            response = self.chatbot.get_response(question)
            response_time = time.time() - start
            
            generation.end(
                output=response,
                usage={
                    "total_tokens": len(response.split()),
                    "response_time_seconds": response_time
                }
            )
            
            # Update with output
            trace.update(
                output=response,
                metadata={
                    "response_time": response_time
                }
            )
            
            # Check handling
            handled_well = any(
                phrase in response.lower() 
                for phrase in ["carro", "vehicle", "i can help", "car-related"]
            )
            
            trace.score(
                name="handled_appropriately",
                value=1.0 if handled_well else 0.0
            )
            
            print(f"✅ Handled appropriately: {handled_well}")
            
        except Exception as e:
            generation.end(
                level="ERROR",
                status_message=str(e)
            )
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("☁️  Connecting to Langfuse Cloud...")
    print("📍 Region: US (https://us.cloud.langfuse.com)")
    print("")
    
    try:
        bot = CarroChatbot()
        tester = ChatbotTesterWithLangfuse(bot)
        tester.run_simple_tests()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your Langfuse Cloud API keys in secrets.toml")
        print("   2. Ensure you have internet connectivity")
        print("   3. Verify the PDF path exists")

# class CarroChatbot:
#     def __init__(self):
#         pdf_path = "/Users/yuhantey/Downloads/carro-chatbot/faq/carro_malaysia_terms_of_use.pdf"
#         self.index, self.chunk_texts, self.openai_client = init_faq_system(pdf_path)
#         if not all([self.index, self.chunk_texts, self.openai_client]):
#             raise Exception("Failed to initialize FAQ system")

#     def get_response(self, question: str) -> str:
#         """Wrapper method to match the test interface"""
#         response = get_answer_from_faq(question, self.index, self.openai_client, self.chunk_texts)
#         return response.message

# class ChatbotTesterWithLangfuse:
#     def __init__(self, chatbot):
#         self.chatbot = chatbot
#         self.langfuse = Langfuse(
#             host=LANGFUSE_HOST,
#             public_key=LANGFUSE_PUBLIC_KEY,
#             secret_key=LANGFUSE_SECRET_KEY
#         )
        
#     def run_simple_tests(self):
#         """Run basic tests with local Langfuse tracking"""
#         print("🧪 Testing Chatbot with Local Langfuse...")
#         print(f"📍 Langfuse URL: {LANGFUSE_HOST}")
#         print("=" * 50 + "\n")
        
#         self.test_faq()
#         self.test_search()
#         self.test_out_of_scope()
        
#         print(f"\n✅ Tests complete!")
#         print(f"📊 View results at: {LANGFUSE_HOST}")
        
#     def test_faq(self):
#         """Test FAQ question"""
#         question = "What services does Carro provide?"
#         print(f"📋 Testing FAQ: {question}")
        
#         trace = self.langfuse.trace(
#             name="test-faq",
#             input=question,
#             metadata={
#                 "test_type": "FAQ",
#                 "timestamp": datetime.now().isoformat()
#             }
#         )
        
#         generation = trace.generation(
#             name="gemini-response",
#             model="gemini-1.5-flash",  # Use your actual model name
#             input=question
#         )
        
#         start = time.time()
#         response = self.chatbot.get_response(question)
#         response_time = time.time() - start
        
#         generation.end(
#             output=response,
#             usage={
#                 "total_tokens": len(response.split()),
#                 "response_time_seconds": response_time
#             }
#         )
        
#         # Update trace with output
#         trace.update(
#             output=response,
#             metadata={
#                 "response_time": response_time,
#                 "response_length": len(response)
#             }
#         )
        
#         # Score the test
#         keywords = ["sell", "buy", "vehicle", "service"]
#         found = sum(1 for k in keywords if k in response.lower())
#         accuracy = found / len(keywords)
        
#         trace.score(
#             name="accuracy",
#             value=accuracy,
#             comment=f"Found {found}/{len(keywords)} keywords"
#         )
        
#         print(f"✅ Response in {response_time:.2f}s - Accuracy: {accuracy*100:.0f}%")
        
#     def test_search(self):
#         """Test search question"""
#         question = "What is the current price of Mazda CX-5 in Malaysia?"
#         print(f"\n🔍 Testing Search: {question}")
        
#         trace = self.langfuse.trace(
#             name="test-search",
#             input=question,
#             metadata={"test_type": "search"}
#         )
        
#         generation = trace.generation(
#             name="gemini-search",
#             model="gemini-1.5-flash",  # Use your actual model name
#             input=question
#         )
        
#         start = time.time()
#         response = self.chatbot.get_response(question)
#         response_time = time.time() - start
        
#         generation.end(output=response)
        
#         # Update trace with output
#         trace.update(
#             output=response,
#             metadata={
#                 "response_time": response_time,
#                 "has_price": "RM" in response
#             }
#         )
        
#         # Check for price info
#         has_price = "RM" in response or "price" in response.lower()
#         trace.score(
#             name="has_price",
#             value=1.0 if has_price else 0.0
#         )
        
#         print(f"✅ Response in {response_time:.2f}s - Has price: {has_price}")
        
#     def test_out_of_scope(self):
#         """Test out-of-scope question"""
#         question = "What's the weather today?"
#         print(f"\n🚫 Testing Out-of-scope: {question}")
        
#         trace = self.langfuse.trace(
#             name="test-out-of-scope",
#             input=question,
#             metadata={"test_type": "out_of_scope"}
#         )
        
#         start = time.time()
#         response = self.chatbot.get_response(question)
#         response_time = time.time() - start
        
#         # Update with output
#         trace.update(
#             output=response,
#             metadata={
#                 "response_time": response_time
#             }
#         )
        
#         # Check handling
#         handled_well = any(
#             phrase in response.lower() 
#             for phrase in ["carro", "vehicle", "i can help", "car-related"]
#         )
        
#         trace.score(
#             name="handled_appropriately",
#             value=1.0 if handled_well else 0.0
#         )
        
#         print(f"✅ Handled appropriately: {handled_well}")

# if __name__ == "__main__":
#     print("🐳 Make sure Langfuse Docker is running:")
#     print("   docker-compose up -d")
#     print("")
    
#     try:
#         bot = CarroChatbot()
#         tester = ChatbotTesterWithLangfuse(bot)
#         tester.run_simple_tests()
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")