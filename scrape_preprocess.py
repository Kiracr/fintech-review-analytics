# scrape_preprocess.py
import pandas as pd
import numpy as np
import time
from google_play_scraper import app, Sort, reviews
import logging
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the banks and their Google Play App IDs
# CRITICAL: Verify these IDs on the play store if errors occur (https://play.google.com/store/apps/details?id=...)
APP_DEFINITIONS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
     "Bank of Abyssinia": "com.boa.boaMobileBanking",
     "Dashen Bank": "com.cr2.amolelight",
}
# Target slightly more to account for cleaning
TARGET_COUNT_PER_BANK = 500 
MIN_REQUIRED_TOTAL = 1200
OUTPUT_FILENAME = "ethiopian_bank_reviews_cleaned.csv"
COUNTRY = 'us' # 'us' often yields more reviews than 'et', adjust if needed
LANGUAGE = 'en' # filter for english, though some Amharic might slip through or non-lang text
COLUMNS_MAP = {
        'content': 'review',
        'score': 'rating',
        'at': 'date',
      }
FINAL_COLUMNS = ['review', 'rating', 'date', 'bank', 'source']

# --- Scraping Function ---

def scrape_bank_reviews(bank_name, app_id, count, lang, country):
    """Scrapes reviews for a single bank app."""
    logging.info(f"--- Starting scrape for: {bank_name} ({app_id}) ---")
    try:
        # Fetch reviews
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,       # Get the most recent
            count=count,             # Number of reviews to try and retrieve
            filter_score_with=None   # Get all ratings
        )
        
        if not result:
             logging.warning(f"No reviews found for {bank_name} ({app_id}). Check ID, country, language.")
             return pd.DataFrame()

        df = pd.DataFrame(result)
        df['bank'] = bank_name
        df['source'] = 'Google Play Store'
        logging.info(f"Successfully scraped {len(df)} reviews for {bank_name}.")
        # Add a small pause to be polite to the server
        time.sleep(2) 
        return df

    except Exception as e:
        logging.error(f"ERROR scraping {bank_name} ({app_id}): {e}")
        return pd.DataFrame() # Return empty df on error