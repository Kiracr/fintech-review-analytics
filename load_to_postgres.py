# load_to_postgres.py

import pandas as pd
import psycopg2
import logging
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---
# !! IMPORTANT !!
# UPDATE THESE DETAILS TO MATCH YOUR LOCAL POSTGRESQL SETUP
# ---
DB_CONFIG = {
    "dbname": "bank_reviews", # The database you created
    "user": "bank_user",     # The user you created
    "password": "123456789",  # The password you set for the user
    "host": "localhost",     # Usually 'localhost' for a local installation
    "port": "5432"           # Default PostgreSQL port
}

INPUT_CSV = "ethiopian_bank_reviews_analyzed.csv"

#
# --- The rest of the script (create_schema, insert_data, main) remains exactly the same ---
# ... (copy the rest of the script from the previous answer)
#

def create_schema(conn):
    """Creates the database schema (tables and relationships)."""
    create_banks_table = """
    CREATE TABLE IF NOT EXISTS banks (
        bank_id SERIAL PRIMARY KEY,
        bank_name VARCHAR(100) NOT NULL UNIQUE
    );
    """
    create_reviews_table = """
    CREATE TABLE IF NOT EXISTS reviews (
        review_id SERIAL PRIMARY KEY,
        bank_id INTEGER NOT NULL,
        review_text TEXT,
        rating INTEGER,
        review_date DATE,
        sentiment_label VARCHAR(10),
        sentiment_score NUMERIC(10, 9),
        themes TEXT,
        CONSTRAINT fk_bank
            FOREIGN KEY(bank_id) 
            REFERENCES banks(bank_id)
            ON DELETE CASCADE
    );
    """
    try:
        with conn.cursor() as cur:
            logging.info("Creating 'banks' table if it doesn't exist...")
            cur.execute(create_banks_table)
            logging.info("Creating 'reviews' table if it doesn't exist...")
            cur.execute(create_reviews_table)
        conn.commit()
        logging.info("Schema created successfully.")
    except Exception as e:
        logging.error(f"Error creating schema: {e}")
        conn.rollback()
        raise e

def insert_data(conn, df):
    """Inserts cleaned data from a DataFrame into the database."""
    try:
        with conn.cursor() as cur:
            # --- Step 1: Insert unique banks and get their IDs ---
            logging.info("Inserting unique bank names...")
            unique_banks = df['bank'].unique()
            
            # Use ON CONFLICT to avoid errors if banks already exist
            insert_bank_sql = "INSERT INTO banks (bank_name) VALUES (%s) ON CONFLICT (bank_name) DO NOTHING;"
            for bank_name in unique_banks:
                cur.execute(insert_bank_sql, (bank_name,))
            
            # Fetch the bank IDs into a dictionary for easy lookup
            cur.execute("SELECT bank_id, bank_name FROM banks;")
            bank_id_map = {name: id for id, name in cur.fetchall()}
            logging.info(f"Bank ID map: {bank_id_map}")

            # --- Step 2: Insert reviews ---
            logging.info(f"Inserting {len(df)} reviews into the database...")
            
            insert_review_sql = """
            INSERT INTO reviews (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, themes)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            
            for _, row in df.iterrows():
                bank_id = bank_id_map.get(row['bank'])
                if bank_id is None:
                    logging.warning(f"Skipping row with unknown bank: {row['bank']}")
                    continue
                
                # Prepare a tuple of values for insertion
                review_data = (
                    bank_id,
                    row['review'],
                    row['rating'],
                    row['date'],
                    row['sentiment_label'],
                    row['sentiment_score'],
                    row['theme']
                )
                cur.execute(insert_review_sql, review_data)
        
        # Commit the transaction if all insertions are successful
        conn.commit()
        logging.info("Data inserted successfully.")

    except Exception as e:
        logging.error(f"Error during data insertion: {e}")
        # Roll back the transaction in case of an error
        conn.rollback()
        raise e

# --- Main Execution ---

def main():
    """Main function to orchestrate the database loading process."""
    # 1. Load Data from CSV
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}. Please run Task 2 first.")
        return
    
    logging.info(f"Loading analyzed data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    # Drop rows where essential data might be null before DB insertion
    df.dropna(subset=['bank', 'review', 'rating', 'date'], inplace=True)
    
    # 2. Connect to Database and Load Data
    conn = None
    try:
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Create schema
        create_schema(conn)
        
        # Insert data
        insert_data(conn, df)
        
    except psycopg2.OperationalError as e:
        logging.error(f"DATABASE CONNECTION FAILED: {e}")
        logging.error("Please ensure your local PostgreSQL server is running and the connection details in DB_CONFIG are correct.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()