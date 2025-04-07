from fastapi import FastAPI, Query, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Union
import pandas as pd
import json
import os
from pydantic import BaseModel, Field, HttpUrl
from embeddings import EmbeddingIndex, extract_text_from_url
from scraper import scrape_shl_catalog, load_catalog_data

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API to recommend SHL assessments based on job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding index
index_path = "data/faiss_index"
index = None

# Define response models
class AssessmentRecommendation(BaseModel):
    name: str
    url: Optional[str] = None
    test_type: Optional[str] = None
    duration_minutes: Optional[int] = None
    duration_text: Optional[str] = None
    remote_support: str = "No"
    adaptive_support: str = "No"
    score: float
    description: Optional[str] = None
    skills_tested: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[AssessmentRecommendation]
    query: str

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding index on startup"""
    global index
    
    # Check if catalog data exists
    catalog_path = "data/shl_catalog.csv"
    if not os.path.exists(catalog_path):
        # Scrape data if it doesn't exist
        df = scrape_shl_catalog()
        df.to_csv(catalog_path, index=False)
    
    # Check if index exists
    if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}_metadata.pkl"):
        # Load existing index
        index = EmbeddingIndex.load(index_path)
    else:
        # Build index
        index = EmbeddingIndex()
        df = load_catalog_data()
        documents = df.to_dict(orient='records')
        index.build_index(documents, index_path)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "message": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "Get assessment recommendations based on job description",
            "/catalog": "Get all assessments in the catalog"
        }
    }

@app.get("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(
    query: Optional[str] = Query(None, description="Job description text"),
    url: Optional[str] = Query(None, description="URL of job description"),
    k: int = Query(10, description="Number of recommendations to return", ge=1, le=50)
):
    """Get assessment recommendations based on job description"""
    global index
    
    # Ensure index is loaded
    if index is None:
        raise HTTPException(status_code=500, detail="Embedding index not initialized")
    
    # Process query
    if query:
        text = query
    elif url:
        text = extract_text_from_url(url)
    else:
        raise HTTPException(status_code=400, detail="Either 'query' or 'url' parameter is required")
    
    # Get recommendations
    results = index.search(text, k=k)
    
    # Format response
    recommendations = [
        AssessmentRecommendation(
            name=result.get("name", ""),
            url=result.get("url", None),
            test_type=result.get("test_type", None),
            duration_minutes=result.get("duration_minutes", None),
            duration_text=result.get("duration_text", None),
            remote_support=result.get("remote_support", "No"),
            adaptive_support=result.get("adaptive_support", "No"),
            score=result.get("score", 0.0),
            description=result.get("description", None),
            skills_tested=result.get("skills_tested", None)
        )
        for result in results
    ]
    
    return RecommendationResponse(recommendations=recommendations, query=text[:500] + "..." if len(text) > 500 else text)

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_post(
    query: Optional[str] = Form(None, description="Job description text"),
    url: Optional[str] = Form(None, description="URL of job description"),
    file: Optional[UploadFile] = File(None, description="Job description file"),
    k: int = Form(10, description="Number of recommendations to return", ge=1, le=50)
):
    """Get assessment recommendations based on job description (POST method)"""
    # Handle file upload
    text = None
    if file:
        content = await file.read()
        text = content.decode("utf-8")
    elif query:
        text = query
    elif url:
        text = extract_text_from_url(url)
    else:
        raise HTTPException(status_code=400, detail="Either 'query', 'url', or file upload is required")
    
    # Use the same logic as the GET endpoint
    return await recommend(query=text, url=None, k=k)

@app.get("/catalog", tags=["Catalog"])
async def get_catalog():
    """Get all assessments in the catalog"""
    try:
        df = load_catalog_data()
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving catalog: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)