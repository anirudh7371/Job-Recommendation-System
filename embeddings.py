import os
import numpy as np
import pandas as pd
import faiss
import pickle
from sklearn.preprocessing import normalize
import json
from typing import List, Dict, Optional
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

class EmbeddingModel:
    """Base class for embedding models"""

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

class GeminiEmbeddingModel(EmbeddingModel):
    """Embedding model using Gemini API"""

    def __init__(self, model_name: str = "models/embedding-001"):
        genai.api_key = os.getenv("GEMINI_API_KEY")
        if not genai.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(response["embedding"])
        except Exception as e:
            raise RuntimeError(f"Failed to embed text using Gemini: {e}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return np.array(embeddings)

class EmbeddingIndex:
    """Class to manage embedding index for similarity search"""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or GeminiEmbeddingModel()
        self.index = None
        self.documents = []
        self.embedding_dim = None

    def _prepare_text(self, doc: Dict) -> str:
        text_parts = []
        if "name" in doc and doc["name"]:
            text_parts.append(f"Name: {doc['name']}")
        if "test_type" in doc and doc["test_type"]:
            text_parts.append(f"Test Type: {doc['test_type']}")
        if "skills_tested" in doc and doc["skills_tested"]:
            text_parts.append(f"Skills Tested: {doc['skills_tested']}")
        if "description" in doc and doc["description"]:
            text_parts.append(f"Description: {doc['description']}")
        if "duration_text" in doc and doc["duration_text"]:
            text_parts.append(f"Duration: {doc['duration_text']}")
        if "remote_support" in doc:
            text_parts.append(f"Remote Testing: {doc['remote_support']}")
        if "adaptive_support" in doc:
            text_parts.append(f"Adaptive Testing: {doc['adaptive_support']}")
        return " ".join(text_parts)

    def build_index(self, documents: List[Dict], save_path: str = "data/faiss_index"):
        self.documents = documents
        texts = [self._prepare_text(doc) for doc in documents]
        embeddings = self.embedding_model.embed_batch(texts)
        embeddings = normalize(embeddings, axis=1, norm='l2')
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.save(save_path)
        return self

    def search(self, query: str, k: int = 10) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
        query_embedding = self.embedding_model.embed_text(query)
        query_embedding = normalize(query_embedding.reshape(1, -1), axis=1, norm='l2')
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                result = self.documents[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        return results

    def save(self, path: str = "data/faiss_index"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embedding_dim': self.embedding_dim
            }, f)
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str = "data/faiss_index", embedding_model: Optional[EmbeddingModel] = None):
        index_manager = cls(embedding_model or GeminiEmbeddingModel())
        index_manager.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            index_manager.documents = metadata['documents']
            index_manager.embedding_dim = metadata['embedding_dim']
        print(f"Index loaded from {path}")
        return index_manager

def extract_text_from_url(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        return f"Error extracting text from URL: {str(e)}"

def build_embedding_index(catalog_data_path: str = "data/shl_catalog.csv",
                          index_path: str = "data/faiss_index",
                          embedding_model_type: str = "gemini"):
    df = pd.read_csv(catalog_data_path)
    documents = df.to_dict(orient='records')
    model = GeminiEmbeddingModel()
    index_manager = EmbeddingIndex(model)
    index_manager.build_index(documents, index_path)
    return index_manager

if __name__ == "__main__":
    build_embedding_index(
        catalog_data_path="data/shl_catalog.csv",
        index_path="data/faiss_index",
        embedding_model_type="gemini"
    )