# SHL Assessment Recommender

This GenAI-powered application recommends SHL assessments based on job descriptions using NLP and semantic similarity techniques. It supports both web and API access.

---

## 🔗 Live Links

- 🖥️ **Streamlit Web App**: [https://shl-advisor.streamlit.app](https://shl-advisor.streamlit.app)  
- ⚙️ **API Endpoint**: [https://shl-api-j8jp.onrender.com](https://shl-api-j8jp.onrender.com)  
- 📄 **Technical Document**: [SHL_Assessment_Recommender_Tech_Approach.docx](https://drive.google.com/file/d/1tVQsLEPr9xdx6uZ_-1PQf_lXxr0xdBgB/view?usp=share_link)

---

## ✨ Features

- Input job descriptions via **text**, **URL**, or **file upload**
- Get up to **10 personalized** SHL assessment recommendations
- View details like **test type**, **remote support**, and **adaptive/IRT availability**
- Easy-to-use **Streamlit UI** and **REST API**

---

## 🧱 Architecture

1. **Data Collection**: Web scraping SHL catalog  
2. **Embedding Generation**: Convert assessments to vector embeddings (Gemini or Sentence Transformers)  
3. **Vector Search**: Use FAISS for semantic similarity  
4. **Web Interface**: Streamlit for frontend  
5. **REST API**: FastAPI for backend access  

---

## 🚀 Getting Started

### 📦 Installation

```bash
git clone https://github.com/anirudh7371/SHL-Assessment-Recommendation-System.git
cd SHL-Assessment-Recommendation-System
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 🔐 Environment Variables

```bash
# Required for Gemini embeddings
export GEMINI_API_KEY="your_gemini_api_key"
```

---

## 🛠️ Usage

### 📘 Run the Web App

```bash
streamlit run app.py
```
App runs at: `http://localhost:8501`

---

### ⚙️ Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🌐 API Endpoints

### 🔍 Health Check

```
GET /health
→ { "status": "healthy" }
```

---

### 🔎 Recommendations

#### POST `/recommend` (SHL-Compliant)

```bash
POST https://shl-api-j8jp.onrender.com/recommend
Content-Type: application/json

{
  "query": "machine learning engineer with Python experience"
}
```

#### Response Format

```json
{
  "recommended_assessments": [
    {
      "url": "https://example.com",
      "adaptive_support": "Yes",
      "description": "...",
      "duration": 60,
      "remote_support": "No",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

---

### 📚 Catalog

```
GET /catalog
→ Returns all available assessments from the SHL catalog
```

---

## 🗂️ Project Structure

```
📁 SHL-Assessment-Recommender/
├── app.py                 # Streamlit UI
├── api.py                 # FastAPI backend
├── embeddings.py          # Embedding & FAISS logic
├── scraper.py             # SHL catalog scraper
├── data/
│   ├── shl_catalog.csv
│   ├── faiss_index.faiss
│   └── faiss_index_metadata.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Technical Stack

| Layer           | Tech                            |
|----------------|----------------------------------|
| Embeddings      | Gemini API / Sentence Transformers |
| Vector Store    | FAISS (Cosine Similarity)        |
| Web Scraping    | BeautifulSoup                    |
| Backend API     | FastAPI                          |
| Frontend UI     | Streamlit                        |

---

## 📈 Future Enhancements

- Add user feedback loop for improved recommendations
- Support hybrid (keyword + semantic) search
- Provide explainability on why assessments were matched
- Add authentication and rate-limiting for API usage

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- **SHL** for the product catalog
- **Hugging Face** for pretrained embeddings
- **Meta Research** for FAISS
- **Google GenAI** for Gemini embeddings
