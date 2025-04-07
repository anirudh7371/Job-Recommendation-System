# SHL Assessment Recommender: Technical Approach

## Problem Statement
Create a GenAI-powered web application that takes job descriptions (text, URL, or file) and returns relevant SHL assessment recommendations with key information including remote testing and adaptive/IRT support.

## Solution Architecture

### Data Acquisition
- **Source**: SHL Product Catalog (https://www.shl.com/solutions/products/product-catalog/)
- **Method**: Web scraping using BeautifulSoup and requests
- **Data Points**: 
  - Assessment name
  - Test type
  - Duration
  - Remote testing support
  - Adaptive/IRT support
  - Skills tested
  - Description
  - URL

### NLP Pipeline
1. **Text Preprocessing**:
   - HTML/document parsing
   - Text normalization
   - Feature extraction

2. **Embedding Generation**:
   - **Model**: Sentence Transformers (all-MiniLM-L6-v2)
   - **Alternative**: OpenAI Embeddings API (configurable)
   - **Process**: Convert assessment descriptions and job descriptions into dense vector representations (768-dimensional vectors)

3. **Vector Storage & Retrieval**:
   - **Technology**: FAISS (Facebook AI Similarity Search)
   - **Index Type**: Flat index with L2 normalization for cosine similarity
   - **Query Processing**: Convert job descriptions to the same vector space and retrieve top-k similar assessments

### Technical Implementation
- **Backend API**: FastAPI
- **Web UI**: Streamlit
- **Deployment**:
  - API: Render / HuggingFace Spaces
  - Web App: Streamlit Cloud
- **Performance**:
  - Average response time: ~1-2 seconds
  - Memory usage: <500MB

## Evaluation Metrics
- **Relevance**: Manual evaluation of recommendations vs. job requirements
- **Recall@10**: Percentage of relevant assessments found in top 10 recommendations
- **Precision@10**: Percentage of top 10 recommendations that are relevant

## Implementation Details

### Embedding Strategy
Combined multiple fields into a single text representation for more comprehensive matching:
```python
text = f"Name: {assessment['name']} Test Type: {assessment['test_type']} Skills: {assessment['skills_tested']} Description: {assessment['description']}"
```

### Similarity Computation
Used cosine similarity between normalized vectors to find closest matches:
```python
# Normalize vectors
embeddings = normalize(embeddings, axis=1, norm='l2')
# Use dot product for cosine similarity with normalized vectors
index = faiss.IndexFlatIP(embedding_dim)
```

### Advantages of Approach
1. **Semantic Understanding**: Captures meaning beyond keyword matching
2. **Efficiency**: Fast retrieval (sub-second) even with thousands of assessments
3. **Extensibility**: Easy to add new assessments or update existing ones
4. **Deployment Simplicity**: Minimal dependencies, runs well on free-tier hosting

## Future Improvements
1. **Domain Adaptation**: Fine-tune embedding model on HR/recruitment text
2. **Hybrid Search**: Combine semantic search with keyword matching
3. **Explanation**: Add rationale for why each assessment was recommended
4. **User Feedback Loop**: Incorporate user feedback to improve recommendations