import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def scrape_shl_catalog(delay=1.5):
    base_url = "https://www.shl.com/solutions/products/product-catalog/"
    headers = {"User-Agent": "Mozilla/5.0"}
    products = []
    start = 0  # Starting index for pagination

    while True:
        logging.info(f"Scraping SHL product catalog - Starting at index {start}...")
        try:
            response = requests.get(f"{base_url}?start={start}&type=1&type=2", headers=headers)
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Request failed at index {start}: {e}")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.select_one(".custom__table-wrapper table")
        if not table:
            logging.info("No table found - possibly no more pages.")
            break

        rows = table.find_all("tr")[1:]
        if not rows:
            logging.info("No more products found.")
            break

        for row in tqdm(rows, desc="Scraping products"):
            cols = row.find_all("td")
            if len(cols) != 4:
                continue

            product = {}

            # Name and URL
            name_elem = cols[0].find("a")
            product["name"] = name_elem.text.strip() if name_elem else ""
            product["url"] = "https://www.shl.com" + name_elem["href"] if name_elem and name_elem.get("href") else ""

            # Remote/Adaptive Support
            remote_span = cols[1].find("span")
            product["remote_support"] = "Yes" if remote_span and "-yes" in remote_span.get("class", []) else "No"

            adaptive_span = cols[2].find("span")
            product["adaptive_support"] = "Yes" if adaptive_span and "-yes" in adaptive_span.get("class", []) else "No"

            # Test Types
            types = cols[3].find_all("span", class_="product-catalogue__key")
            product["test_types"] = [t.text.strip() for t in types]

            # Product page for more info
            if product["url"]:
                try:
                    time.sleep(delay)
                    detail_response = requests.get(product["url"], headers=headers)
                    detail_response.raise_for_status()
                    detail_soup = BeautifulSoup(detail_response.content, "html.parser")

                    # Description
                    desc_elem = (
                        detail_soup.select_one(".product-catalogue__description p") or
                        detail_soup.find("div", class_="product-catalogue__description") or
                        detail_soup.select_one("meta[name='description']")
                    )
                    if desc_elem:
                        product["description"] = (
                            desc_elem.get("content").strip()
                            if desc_elem.name == "meta"
                            else desc_elem.get_text(strip=True)
                        )
                    else:
                        product["description"] = ""

                    # Features
                    features = detail_soup.select(".product-catalogue__features li")
                    for feat in features:
                        strong = feat.find("strong")
                        if strong:
                            key = (
                                strong.text.strip().rstrip(":")
                                .lower().replace(" ", "_").replace("-", "_")
                            )
                            value = feat.get_text(strip=True).replace(strong.text, "").strip()
                            product[key] = value
                except Exception as e:
                    logging.warning(f"Failed to scrape inner page for {product['name']}: {e}")
                    product["description"] = ""

            products.append(product)

        start += len(rows)

    logging.info(f"Scraped {len(products)} products.")
    return pd.DataFrame(products)

def save_catalog_data(df, csv_path="data/shl_catalog.csv", json_path="data/shl_catalog.json"):
    """Saving the catalog data to CSV and JSON files"""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    logging.info(f"Saved catalog data to {csv_path}")

    with open(json_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)
    logging.info(f"Saved catalog data to {json_path}")

def load_catalog_data(csv_path="data/shl_catalog.csv"):
    """Loading the catalog data from CSV file"""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_path}")
        return None

if __name__ == "__main__":
    csv_path = "data/shl_catalog.csv"
    if os.path.exists(csv_path):
        logging.info(f"Catalog data already exists at {csv_path}")
        df = load_catalog_data(csv_path)
        logging.info(f"Loaded {len(df)} products from existing data")
    else:
        df = scrape_shl_catalog()
        if df is not None and not df.empty:
            save_catalog_data(df)
            logging.info("Catalog data scraped and saved successfully")
        else:
            logging.error("Failed to scrape catalog data")
