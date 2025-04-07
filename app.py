import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO
from typing import Optional, Dict, List, Any
import os
import sys
from embeddings import GeminiEmbeddingModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from embeddings import EmbeddingIndex, extract_text_from_url
    from scraper import load_catalog_data
    local_api = True
except ImportError:
    local_api = False

# Set page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_K = 10


if "results" not in st.session_state:
    st.session_state.results = None
if "query" not in st.session_state:
    st.session_state.query = ""


def get_recommendations_from_api(query: str = None, url: str = None, file_content: str = None, k: int = DEFAULT_K) -> Dict:
    """Get recommendations from the API"""
    try:
        if query:
            response = requests.get(f"{API_URL}/recommend", params={"query": query, "k": k})
        elif url:
            response = requests.get(f"{API_URL}/recommend", params={"url": url, "k": k})
        elif file_content:
            response = requests.post(
                f"{API_URL}/recommend",
                data={"k": k},
                files={"file": ("job_description.txt", file_content)}
            )
        else:
            st.error("No input provided")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_recommendations_local(query: str = None, url: str = None, file_content: str = None, k: int = DEFAULT_K) -> Dict:
    """Get recommendations using local modules"""
    try:
        # Ensure index is loaded
        index_path = "data/faiss_index"
        if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}_metadata.pkl"):
            index = EmbeddingIndex.load(index_path)
        else:
            model = GeminiEmbeddingModel()
            index = EmbeddingIndex(model)
            df = load_catalog_data()
            documents = df.to_dict(orient='records')
            index.build_index(documents, index_path)
        
        # Process query
        if query:
            text = query
        elif url:
            text = extract_text_from_url(url)
        elif file_content:
            text = file_content
        else:
            st.error("No input provided")
            return None
        
        # Get recommendations
        results = index.search(text, k=k)
        
        # Format response similar to API
        return {
            "recommendations": results,
            "query": text[:500] + "..." if len(text) > 500 else text
        }
    except Exception as e:
        st.error(f"Error getting recommendations locally: {str(e)}")
        return None

def get_recommendations(query: str = None, url: str = None, file_content: str = None, k: int = DEFAULT_K) -> Dict:
    """Get recommendations either from API or local modules"""
    if local_api:
        return get_recommendations_local(query, url, file_content, k)
    else:
        return get_recommendations_from_api(query, url, file_content, k)

def display_recommendations(results: Dict):
    """Display recommendations in a formatted way"""
    if not results or "recommendations" not in results:
        st.warning("No recommendations found")
        return
    
    recommendations = results["recommendations"]
    
    if not recommendations:
        st.warning("No matching assessments found")
        return
    
    df = pd.DataFrame(recommendations)
    
    st.subheader(f"Top {len(recommendations)} Recommended Assessments")
    
    display_df = df.copy()
    
    if "url" in display_df.columns and "name" in display_df.columns:
        display_df["name"] = display_df.apply(
            lambda row: f"[{row['name']}]({row['url']})" if pd.notna(row['url']) else row['name'], 
            axis=1
        )
    
    display_columns = [
        "name", "test_type", "duration_text", "remote_support", 
        "adaptive_support", "score"
    ]
    
    display_columns = [col for col in display_columns if col in display_df.columns]
    
    column_rename = {
        "name": "Assessment Name",
        "test_type": "Test Type",
        "duration_text": "Duration",
        "remote_support": "Remote Testing",
        "adaptive_support": "Adaptive/IRT",
        "score": "Match Score"
    }
    
    rename_map = {col: column_rename[col] for col in display_columns if col in column_rename}
    display_df = display_df[display_columns].rename(columns=rename_map)
    
    if "Match Score" in display_df.columns:
        display_df["Match Score"] = display_df["Match Score"].apply(lambda x: f"{x:.2f}")
    
    st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
    
    st.subheader("Detailed Assessment Information")
    
    cols = st.columns(3)
    for i, rec in enumerate(recommendations):
        with cols[i % 3]:
            with st.expander(f"{i+1}. {rec['name']}", expanded=False):
                if "url" in rec and rec["url"]:
                    st.markdown(f"[View Assessment]({rec['url']})")
                
                if "test_type" in rec and rec["test_type"]:
                    st.markdown(f"**Type:** {rec['test_type']}")
                
                if "duration_text" in rec and rec["duration_text"]:
                    st.markdown(f"**Duration:** {rec['duration_text']}")
                
                st.markdown(f"**Remote Testing:** {rec.get('remote_support', 'No')}")
                st.markdown(f"**Adaptive/IRT:** {rec.get('adaptive_support', 'No')}")
                
                if "description" in rec and rec["description"]:
                    st.markdown(f"**Description:** {rec['description']}")
                
                if "skills_tested" in rec and rec["skills_tested"]:
                    st.markdown(f"**Skills Tested:** {rec['skills_tested']}")
                
                st.markdown(f"**Match Score:** {rec.get('score', 0):.2f}")

# App header
st.title("SHL Assessment Recommender")
st.markdown("""
This app recommends SHL assessments based on job descriptions. Enter a job description, 
provide a URL, or upload a file to get personalized assessment recommendations.
""")

# Sidebar
with st.sidebar:
    st.header("Options")
    k = st.slider("Number of recommendations", min_value=1, max_value=20, value=DEFAULT_K)
    
    st.header("About")
    st.markdown("""
    This app uses natural language processing and embeddings to match job descriptions 
    with the most relevant SHL assessments.
    
    **How it works:**
    1. Each assessment in the SHL catalog is converted to an embedding vector
    2. Your job description is also converted to an embedding
    3. The system finds assessments with the most similar embeddings
    
    **Tech Stack:**
    - Streamlit for the UI
    - FastAPI for the backend
    - Gemini API for embeddings
    - FAISS for similarity search
    """)

# Input method tabs
tab1, tab2, tab3 = st.tabs(["Enter Text", "Provide URL", "Upload File"])

with tab1:
    query = st.text_area("Enter job description", height=200, key="text_input")
    if st.button("Get Recommendations", key="text_button"):
        if query:
            with st.spinner("Getting recommendations..."):
                st.session_state.results = get_recommendations(query=query, k=k)
                st.session_state.query = query
        else:
            st.error("Please enter a job description")

with tab2:
    url = st.text_input("Enter job posting URL", key="url_input")
    if st.button("Get Recommendations", key="url_button"):
        if url:
            with st.spinner("Extracting text and getting recommendations..."):
                st.session_state.results = get_recommendations(url=url, k=k)
                st.session_state.query = f"URL: {url}"
        else:
            st.error("Please enter a URL")

with tab3:
    uploaded_file = st.file_uploader("Upload File", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_content = None
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                from io import BytesIO
                
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                file_content = ""
                for page_num in range(len(pdf_reader.pages)):
                    file_content += pdf_reader.pages[page_num].extract_text() + "\n"
            except ImportError:
                st.error("PyPDF2 package not found. Unable to parse PDF files.")
                st.info("Please install PyPDF2 with `pip install PyPDF2` or use text input instead.")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                import docx
                from io import BytesIO
                
                doc = docx.Document(BytesIO(uploaded_file.getvalue()))
                file_content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                st.error("python-docx package not found. Unable to parse DOCX files.")
                st.info("Please install python-docx with `pip install python-docx` or use text input instead.")
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
        
        if file_content:
            st.success(f"File '{uploaded_file.name}' uploaded successfully")
            if st.button("Get Recommendations", key="file_button"):
                with st.spinner("Processing file and getting recommendations..."):
                    st.session_state.results = get_recommendations(file_content=file_content, k=k)
                    st.session_state.query = f"File: {uploaded_file.name}"

# Displaying results
if st.session_state.results:
    st.markdown("---")
    st.header("Results")
    
    # Show query used
    with st.expander("Query Used", expanded=False):
        st.markdown(f"```\n{st.session_state.query}\n```")
    
    # Display recommendations
    display_recommendations(st.session_state.results)

#Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
    SHL Assessment Recommender | Built with ❤️ using Streamlit and NLP
    </div>
    """, 
    unsafe_allow_html=True
)

if __name__ == "__main__":
    pass