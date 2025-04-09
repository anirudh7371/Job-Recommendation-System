from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
from embeddings import EmbeddingIndex
from scraper import scrape_shl_catalog, load_catalog_data

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API to recommend SHL assessments based on job descriptions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index_path = "data/faiss_index"
index = None

class QueryInput(BaseModel):
    query: str

# Response format SHL expects
class Assessment(BaseModel):
    url: Optional[str]
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationAPIResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.on_event("startup")
async def load_index():
    global index
    catalog_path = "data/shl_catalog.csv"
    if not os.path.exists(catalog_path):
        df = scrape_shl_catalog()
        df.to_csv(catalog_path, index=False)
    if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}_metadata.pkl"):
        index = EmbeddingIndex.load(index_path)
    else:
        df = load_catalog_data()
        index = EmbeddingIndex()
        index.build_index(df.to_dict(orient="records"), save_path=index_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationAPIResponse)
async def recommend_api(payload: QueryInput):
    global index
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
    
    query_text = payload.query
    results = index.search(query_text, k=10)

    response = {
        "recommended_assessments": [
            {
                "url": result.get("url", ""),
                "adaptive_support": "Yes" if result.get("adaptive_support") == "Yes" else "No",
                "description": result.get("description", ""),
                "duration": int(result.get("duration_minutes") or 0),
                "remote_support": "Yes" if result.get("remote_support") == "Yes" else "No",
                "test_type": [result.get("test_type")] if result.get("test_type") else []
            }
            for result in results[:10]
        ]
    }

    return response

@app.get("/catalog")
async def get_catalog():
    try:
        df = load_catalog_data()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving catalog: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)