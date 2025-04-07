import os
import argparse
import time
from scraper import scrape_shl_catalog, save_catalog_data, load_catalog_data
from embeddings import build_embedding_index, EmbeddingIndex

def run_pipeline(force_rebuild=False):
    """Run the complete pipeline"""
    start_time = time.time()
    print("=== SHL Assessment Recommender Pipeline ===")

    # Step 1: Check if catalog data exists
    catalog_path = "data/shl_catalog.csv"
    if not os.path.exists(catalog_path) or force_rebuild:
        print("\n== Step 1: Scraping SHL Catalog ==")
        df = scrape_shl_catalog()
        if df is not None and not df.empty:
            save_catalog_data(df)
            print(f"✓ Catalog data scraped and saved successfully ({len(df)} assessments)")
        else:
            print("✗ Failed to scrape catalog data")
            return False
    else:
        print("\n== Step 1: Using existing catalog data ==")
        df = load_catalog_data()
        print(f"✓ Loaded {len(df)} assessments from existing data")

    # Step 2: Build embedding index
    index_path = "data/faiss_index"
    if not os.path.exists(f"{index_path}.faiss") or force_rebuild:
        print("\n== Step 2: Building embedding index ==")
        try:
            build_embedding_index(
                catalog_data_path=catalog_path,
                index_path=index_path,
                embedding_model_type="gemini"  # Currently hardcoded
            )
            print("✓ Embedding index built successfully")
        except Exception as e:
            print(f"✗ Failed to build embedding index: {str(e)}")
            return False
    else:
        print("\n== Step 2: Using existing embedding index ==")
        print("✓ Embedding index already exists")

    # Step 3: Test index with a sample query
    print("\n== Step 3: Testing index with sample query ==")
    try:
        index = EmbeddingIndex.load(index_path)
        sample_query = "Data scientist with Python and machine learning skills"
        print(f"Sample query: '{sample_query}'")

        results = index.search(sample_query, k=3)
        print(f"✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['name']} (Score: {result['score']:.4f})")

    except Exception as e:
        print(f"✗ Failed to test index: {str(e)}")
        return False

    elapsed_time = time.time() - start_time
    print(f"\n✓ Pipeline completed successfully in {elapsed_time:.2f} seconds")
    print("\nNext steps:")
    print("1. Run the API:  uvicorn api:app --host 0.0.0.0 --port 8000")
    print("2. Run the UI:   streamlit run app.py")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SHL Assessment Recommender pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of catalog data and embedding index")
    args = parser.parse_args()

    run_pipeline(force_rebuild=args.rebuild)