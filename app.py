import streamlit as st
import os
import tempfile
from datetime import datetime

# --- LangChain & AI Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# --- Page Config ---
st.set_page_config(page_title="Hybrid RAG Search Engine (Groq)", layout="wide")

# --- Sidebar: Configuration & Data ---
with st.sidebar:
    st.title("Configuration")
    
    # API Keys Input
    groq_api_key = st.text_input("Groq API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.divider()
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    
    process_btn = st.button("Process & Index Documents")

# --- Backend Logic ---

# 1. Router Logic (Classify Query)
class RouteQuery(BaseModel):
    """Route the query to the appropriate data source."""
    datasource: str = Field(
        ..., 
        description="Choose 'vectorstore' for specific internal document questions, 'web_search' for current events/general knowledge, or 'hybrid' for both."
    )

def get_router(llm):
    structured_llm = llm.with_structured_output(RouteQuery)
    return structured_llm

# 2. Document Processing (FIXED for Windows)
def process_documents(uploaded_files):
    """Load, split, and index PDFs using local embeddings."""
    if not uploaded_files:
        return None
    
    all_docs = []
    
    # Handle temp files for Streamlit
    for uploaded_file in uploaded_files:
        # Create a temp file
        # We use delete=False because we need to close it before PyPDFLoader can open it on Windows
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name # Save the path
            
            # --- FILE IS NOW CLOSED ---
            # Now it is safe to open with PyPDFLoader on Windows
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Add metadata source
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["type"] = "document"
            
            all_docs.extend(docs)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Ensure the temp file is deleted even if an error occurs
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    # Create Vector Store (Using Free HuggingFace Embeddings)
    if splits:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    else:
        return None

# 3. Web Search Logic
def perform_web_search(query):
    """Execute Tavily search and format results."""
    tool = TavilySearchResults(k=3)
    try:
        results = tool.invoke({"query": query})
        web_context = []
        # Tavily results sometimes come back as a list of dicts, sometimes raw strings depending on version/errors
        if isinstance(results, list):
            for res in results:
                web_context.append(
                    f"[Web Source: {res.get('url', 'N/A')}]\nContent: {res.get('content', 'N/A')}"
                )
            return "\n\n".join(web_context), results
        else:
            return str(results), []
    except Exception as e:
        return f"Error performing web search: {e}", []

# 4. Main RAG Chain
def hybrid_rag_response(query, vectorstore, llm):
    # Step A: Routing
    router = get_router(llm)
    try:
        route_prediction = router.invoke(f"Route this query: {query}")
        route = route_prediction.datasource
    except Exception as e:
        # Fallback if router fails
        route = "hybrid"
    
    context_text = ""
    sources_used = []

    # Step B: Retrieval
    # 1. Vector Search (if needed)
    if route in ["vectorstore", "hybrid"] and vectorstore:
        docs = vectorstore.similarity_search(query, k=3)
        doc_context = "\n\n".join([f"[Doc: {d.metadata['source']}]\n{d.page_content}" for d in docs])
        context_text += f"--- INTERNAL DOCUMENTS ---\n{doc_context}\n\n"
        sources_used.extend([{"type": "doc", "source": d.metadata["source"], "content": d.page_content} for d in docs])

    # 2. Web Search (if needed)
    if route in ["web_search", "hybrid"]:
        web_text, raw_web_results = perform_web_search(query)
        context_text += f"--- WEB SEARCH RESULTS ---\n{web_text}\n\n"
        sources_used.extend([{"type": "web", "source": r.get('url'), "content": r.get('content')} for r in raw_web_results])

    # Step C: Generation
    system_prompt = """You are a Hybrid RAG assistant using Llama 3. 
    Use the provided context to answer the user's question. 
    
    RULES:
    1. If the answer comes from a document, cite it as [Doc: Filename].
    2. If the answer comes from the web, cite it as [Web: URL].
    3. If you don't know, say so. Do not hallucinate.
    
    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})
    
    return response, sources_used, route

# --- UI Logic ---

st.header("Multi-Document Hybrid RAG")

if not groq_api_key or not tavily_api_key:
    st.warning("Please enter your Groq and Tavily API keys in the sidebar to continue.")
else:
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # Initialize Session State
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle Document Processing
    if process_btn and uploaded_files:
        with st.spinner("Chunking and Indexing Documents (using local embeddings)..."):
            st.session_state.vectorstore = process_documents(uploaded_files)
            if st.session_state.vectorstore:
                st.success("Documents Indexed in FAISS Vector Store!")
            else:
                st.warning("No text could be extracted from the uploaded documents.")

    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for s in msg["sources"]:
                        st.caption(f"**{s['type'].upper()}:** {s['source']}")
                        st.text(s['content'][:150] + "...")

    # Handle User Input
    if user_query := st.chat_input("Ask about your docs or the web..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if not st.session_state.vectorstore and "docs" in user_query.lower():
                    st.warning("No documents indexed! Using web search only.")
                
                response_text, sources, route_taken = hybrid_rag_response(
                    user_query, 
                    st.session_state.vectorstore, 
                    llm
                )
                
                st.markdown(response_text)
                
                # Show Route Indicator
                if route_taken == "web_search":
                    st.caption("Fetched from Web")
                elif route_taken == "vectorstore":
                    st.caption("Fetched from Docs")
                else:
                    st.caption("Hybrid Retrieval (Web + Docs)")

                # Show Sources
                if sources:
                    with st.expander("View Sources"):
                        for s in sources:
                            # icon = "📄" if s['type'] == 'doc' else "🌐"
                            # st.markdown(f"**{icon} {s['source']}**")
                            st.text(s['content'][:200])

        # Save History
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources
        })