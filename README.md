# SHL Assessment Recommender

This application recommends SHL assessments based on job descriptions using natural language processing and semantic search techniques.

## Features

- Input job descriptions via text, URL, or file upload
- Get up to 10 personalized SHL assessment recommendations
- View detailed information for each assessment including remote testing and adaptive/IRT support
- Access both web UI and API endpoints

## Architecture

The application has the following components:

1. **Data Collection**: Scraper for SHL product catalog
2. **Embedding Generation**: Convert assessment descriptions to vector embeddings
3. **Similarity Search**: Find assessments that match job descriptions using FAISS
4. **Web UI**: Streamlit-based user interface
5. **API**: FastAPI endpoints for programmatic access

## Setup Instructions

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SHL-Recommender.git
cd SHL-Recommender
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Set up environment variables:
```bash
# For using OpenAI embeddings (optional)
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Running the Application

#### 1. Data Collection and Indexing

First, run the scraper to collect SHL product data and build the embedding index:

```bash
python scraper.py
```

#### 2. Start the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

#### 3. Launch the Web UI

```bash
streamlit run app.py
```

The Streamlit app will be available at http://localhost:8501

## Usage

### Web UI

1. Access the Streamlit web interface
2. Choose your input method:
   - Enter job description text
   - Provide a URL to a job posting
   - Upload a job description file (TXT, PDF, DOCX)
3. Click "Get Recommendations"
4. View the list of recommended assessments and detailed information

### API

The API provides the following endpoints:

- `GET /recommend?query=...`: Get recommendations based on text query
- `GET /recommend?url=...`: Get recommendations based on URL
- `POST /recommend`: Submit job description as file upload
- `GET /catalog`: Get the full SHL assessment catalog

Example API call:

```bash
curl -X GET "http://localhost:8000/recommend?query=data%20scientist%20with%20python%20skills"
```

## Project Structure

```
📁 SHL-Recommender/
│
├── app.py                # Streamlit App
├── api.py                # FastAPI app
├── embeddings.py         # Embedding + FAISS logic
├── scraper.py            # SHL catalog parser
├── data/
│   ├── shl_catalog.csv   # Scraped assessment data
│   ├── shl_catalog.json  # Scraped data in JSON format
│   ├── faiss_index.faiss # Vector index
│   └── faiss_index_metadata.pkl # Index metadata
├── requirements.txt
└── README.md
```

## Technical Details

- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2) or OpenAI Embeddings
- **Vector Database**: FAISS for efficient similarity search
- **Similarity Metric**: Cosine similarity
- **Data Processing**: BeautifulSoup for web scraping
- **API Framework**: FastAPI
- **UI Framework**: Streamlit

## Future Improvements

- Add user feedback mechanism for recommendations
- Implement more advanced NLP techniques for better matching
- Add support for more file formats
- Explore hybrid search (keyword + semantic)
- Add authentication for API endpoints

## License

MIT License

## Acknowledgements

- SHL for their comprehensive assessment catalog
- HuggingFace for the Sentence Transformers models
- Facebook Research for FAISS