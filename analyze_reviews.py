# analyze_reviews.py
import pandas as pd
import numpy as np
import logging
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress TensorFlow/Hugging Face logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
from transformers import pipeline

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_CSV = "ethiopian_bank_reviews_cleaned.csv"
OUTPUT_CSV = "ethiopian_bank_reviews_analyzed.csv"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Define themes and associated keywords.
# Keywords should be lemmatized (base form of the word) and lowercase.
THEME_KEYWORDS = {
    'Account & Login Issues': ['login', 'password', 'account', 'register', 'access', 'otp', 'block', 'verify', 'verification'],
    'Transaction Performance': ['transfer', 'transaction', 'slow', 'fail', 'stuck', 'error', 'fee', 'charge', 'limit', 'pending'],
    'UI & User Experience': ['ui', 'interface', 'design', 'easy', 'simple', 'update', 'dark', 'mode', 'confuse', 'hard', 'look', 'feel'],
    'Reliability & Bugs': ['crash', 'bug', 'glitch', 'work', 'stop', 'open', 'load', 'freeze', 'problem', 'issue', 'fix'],
    'Customer Support': ['support', 'customer', 'service', 'call', 'center', 'help', 'contact', 'response', 'agent', 'branch'],
    'Features & Functionality': ['feature', 'add', 'option', 'cbebirr', 'telebirr', 'loan', 'statement', 'balance', 'notification']
}

# --- NLP Preprocessing ---
# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

def preprocess_text(text):
    """Lemmatizes and removes stopwords from a text."""
    if not isinstance(text, str):
        return []
    doc = nlp(text.lower())
    lemmas = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return lemmas

# --- Analysis Functions ---

def analyze_sentiment(texts, batch_size=32):
    """
    Analyzes sentiment of a list of texts using a Hugging Face pipeline.
    Uses batching for efficiency.
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, running on CPU. This may be slow.")
        device = -1 # Use CPU
    else:
        logging.info("CUDA is available, running on GPU.")
        device = 0  # Use the first GPU

    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=SENTIMENT_MODEL, 
            device=device
        )
    except Exception as e:
        logging.error(f"Failed to load sentiment model: {e}")
        return None, None

    all_results = []
    logging.info(f"Analyzing sentiment for {len(texts)} reviews...")
    # Process in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_pipeline(batch)
        all_results.extend(results)
        if (i + batch_size) % (batch_size * 5) == 0: # Log progress periodically
            logging.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} reviews...")
    
    labels = [res['label'] for res in all_results]
    scores = [res['score'] for res in all_results]
    return labels, scores

def assign_themes(lemmatized_review):
    """Assigns one or more themes to a review based on keyword matching."""
    assigned_themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in lemmatized_review for keyword in keywords):
            assigned_themes.append(theme)
    
    if not assigned_themes:
        return "General Feedback" # Default theme
    
    return ", ".join(assigned_themes) # Join multiple themes with a comma

def extract_top_keywords_per_theme(df):
    """Uses TF-IDF to find top keywords for each identified theme."""
    logging.info("Extracting top keywords for each theme using TF-IDF...")
    theme_keywords_summary = {}
    
    # We need to handle reviews that have multiple themes. We'll treat each assignment as a document.
    # Explode the DataFrame so each row has one theme.
    df_themes = df.copy()
    df_themes['theme'] = df_themes['theme'].str.split(', ')
    df_exploded = df_themes.explode('theme')
    
    for theme in df_exploded['theme'].unique():
        if theme == "General Feedback":
            continue
            
        # Get all review texts for the current theme
        theme_corpus = df_exploded[df_exploded['theme'] == theme]['review'].tolist()
        
        if len(theme_corpus) < 10: # Skip if not enough data for meaningful TF-IDF
            logging.warning(f"Skipping keyword extraction for theme '{theme}' due to insufficient data ({len(theme_corpus)} reviews).")
            continue
            
        try:
            vectorizer = TfidfVectorizer(max_features=10, preprocessor=lambda x: ' '.join(preprocess_text(x)))
            tfidf_matrix = vectorizer.fit_transform(theme_corpus)
            top_keywords = vectorizer.get_feature_names_out()
            theme_keywords_summary[theme] = top_keywords.tolist()
        except ValueError:
            logging.warning(f"Could not extract keywords for theme '{theme}', likely due to empty vocabulary.")
            theme_keywords_summary[theme] = ["N/A"]

    return theme_keywords_summary

# --- Main Execution ---

def main():
    """Main function to orchestrate the analysis pipeline."""
    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}. Please run Task 1 first.")
        return

    logging.info(f"Loading data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.dropna(subset=['review'], inplace=True) # Ensure no null reviews
    
    # 2. Preprocess Text for Thematic Analysis
    logging.info("Preprocessing text for thematic analysis (lemmatization)...")
    df['lemmatized_review'] = df['review'].apply(preprocess_text)

    # 3. Sentiment Analysis
    reviews_list = df['review'].tolist()
    sentiment_labels, sentiment_scores = analyze_sentiment(reviews_list)
    
    if sentiment_labels is None:
        logging.error("Sentiment analysis failed. Aborting.")
        return
        
    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    # Normalize score: make negative scores negative
    df['sentiment_score'] = df.apply(
        lambda row: -row['sentiment_score'] if row['sentiment_label'] == 'NEGATIVE' else row['sentiment_score'], 
        axis=1
    )
    logging.info("Sentiment analysis complete.")
    
    # 4. Thematic Analysis (Rule-Based)
    logging.info("Assigning themes based on keywords...")
    df['theme'] = df['lemmatized_review'].apply(assign_themes)
    logging.info("Thematic analysis complete.")

    # 5. KPI Reporting & Summary
    print("\n--- Analysis Summary & KPI Check ---")
    
    # KPI 1: Sentiment scores for 90%+ reviews
    sentiment_coverage = (df['sentiment_label'].notna().sum() / len(df)) * 100
    print(f"Sentiment Analysis Coverage: {sentiment_coverage:.2f}% (Target: >90%)")
    if sentiment_coverage > 90:
        print("KPI: Sentiment coverage MET.")
    else:
        print("KPI: Sentiment coverage NOT MET.")

    # Aggregate sentiment by bank
    print("\n--- Average Sentiment Score by Bank (-1 to 1) ---")
    avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values(ascending=False)
    print(avg_sentiment.to_string())

    # Aggregate sentiment by bank and rating
    print("\n--- Average Sentiment by Bank and Rating ---")
    print(df.groupby(['bank', 'rating'])['sentiment_score'].mean().unstack().to_string(float_format="%.2f"))

    # KPI 2: 3+ themes per bank
    print("\n--- Theme Distribution per Bank ---")
    theme_counts = df.groupby('bank')['theme'].value_counts()
    print(theme_counts.to_string())
    
    # KPI 3: Top keywords per theme (validation)
    top_keywords = extract_top_keywords_per_theme(df)
    print("\n--- Top Keywords per Identified Theme (from TF-IDF) ---")
    for theme, keywords in top_keywords.items():
        print(f"  - {theme}: {', '.join(keywords)}")

    # 6. Save Results
    # Select final columns to save
    final_cols = [
        'review', 'rating', 'date', 'bank', 
        'sentiment_label', 'sentiment_score', 'theme'
    ]
    df_to_save = df[final_cols]
    
    logging.info(f"Saving analyzed data to {OUTPUT_CSV}")
    df_to_save.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    logging.info("Analysis pipeline complete.")

if __name__ == "__main__":
    main()