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
    
# --- Preprocessing Function ---

def preprocess_data(df_raw):
     """Cleans and structures the raw scraped data."""
     logging.info("--- Starting Preprocessing ---")
     initial_rows = len(df_raw)
     if initial_rows == 0:
          logging.warning("No raw data to process.")
          return pd.DataFrame(columns=FINAL_COLUMNS), 0, 100.0
          
     logging.info(f"Initial rows fetched: {initial_rows}")
     
     df = df_raw.copy()

     # 1. Handle Missing Critical Data (before calculating missing %)
     # Calculate nulls in key columns BEFORE dropping
     null_content = df['content'].isnull().sum()
     null_score = df['score'].isnull().sum()
     null_date = df['at'].isnull().sum()
     total_nulls = null_content + null_score + null_date
     # Calculate missing percentage based on *any* critical field being null in a row
     rows_with_missing_critical = df[df['content'].isnull() | df['score'].isnull() | df['at'].isnull()]
     missing_percentage = (len(rows_with_missing_critical) / initial_rows) * 100 if initial_rows > 0 else 0
     logging.info(f"Rows with missing critical data (content, score, at): {len(rows_with_missing_critical)} ({missing_percentage:.2f}%)")

    
     # Drop rows where 'content' (review text), score or date is missing - they are unusable
     df.dropna(subset=['content', 'score', 'at'], inplace=True)
      # Also remove empty strings or just whitespace
     df['content'] = df['content'].str.strip()
     df = df[df['content'] != ""]
     df = df[df['content'].str.len() > 0]
     logging.info(f"Rows after dropping null/empty content/score/date: {len(df)}")

      # 2. Remove Duplicates
     # Use reviewId as the most reliable unique identifier from the scraper
     duplicates_count = df.duplicated(subset=['reviewId']).sum()
     df.drop_duplicates(subset=['reviewId'], keep='first', inplace=True)
     logging.info(f"Removed {duplicates_count} duplicate reviews based on reviewId. Rows remaining: {len(df)}")

     # 3. Select and Rename Columns
     # Check if all required columns exist before selecting
     required_raw_cols = list(COLUMNS_MAP.keys()) + ['bank', 'source']
     if not all(col in df.columns for col in required_raw_cols):
          missing = [col for col in required_raw_cols if col not in df.columns]
          logging.error(f"Missing expected columns in dataframe: {missing}")
          return pd.DataFrame(columns=FINAL_COLUMNS), len(df), missing_percentage
          
     df = df[required_raw_cols].rename(columns=COLUMNS_MAP)

     # 4. Normalize Date
     # Convert to datetime first, then format as YYYY-MM-DD string
     df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
     
     # 5. Final Column Order
     df = df[FINAL_COLUMNS]
     
     final_rows = len(df)
     logging.info(f"Preprocessing complete. Final row count: {final_rows}")
     # Verify no nulls remain in the final dataset
     final_null_check = df.isnull().sum().sum()
     if final_null_check > 0:
          logging.warning(f"Warning: {final_null_check} null values detected in final dataset!")
          
     return df, final_rows, missing_percentage

def main():
     """Main function to orchestrate scraping and processing."""
     all_bank_dataframes = []
     
     if os.path.exists(OUTPUT_FILENAME):
        logging.warning(f"Output file '{OUTPUT_FILENAME}' already exists. Overwriting.")

     # Scrape each bank
     for name, app_id in APP_DEFINITIONS.items():
        df_bank = scrape_bank_reviews(name, app_id, TARGET_COUNT_PER_BANK, LANGUAGE, COUNTRY)
        if not df_bank.empty:
             all_bank_dataframes.append(df_bank)
        else:
            logging.warning(f"Skipping {name} due to scraping issues.")

     if not all_bank_dataframes:
          logging.error("No data was scraped for any bank. Exiting.")
          return

     # Combine all data
     df_raw_all = pd.concat(all_bank_dataframes, ignore_index=True)

     # Preprocess
     df_clean, final_count, missing_perc = preprocess_data(df_raw_all)
     
      # Check KPIs
     logging.info("\n--- KPI Check ---")
     logging.info(f"Total Reviews Collected & Cleaned: {final_count} (Target: >={MIN_REQUIRED_TOTAL})")
     logging.info(f"Percentage of rows with missing critical data in raw fetch: {missing_perc:.2f}% (Target: < 5%)")
     if final_count >= MIN_REQUIRED_TOTAL:
         logging.info("KPI: Total review count MET.")
     else:
          logging.warning(f"KPI: Total review count NOT MET. Consider increasing TARGET_COUNT_PER_BANK ({TARGET_COUNT_PER_BANK}) or checking App IDs/Country/Language.")
     if missing_perc < 5.0:
         logging.info("KPI: Missing data percentage MET.")
     else:
          logging.warning("KPI: Missing data percentage NOT MET.")
     
     for name in APP_DEFINITIONS.keys():
         count = len(df_clean[df_clean['bank']==name])
         status = "MET" if count >= 400 else f"NOT MET (Got only {count})"
         logging.info(f"Reviews for {name}: {count} (Target: >=400). Status: {status}")
     logging.info("KPI: Clean CSV dataset & Git Repo organization is manual check.")
     logging.info("-----------------\n")