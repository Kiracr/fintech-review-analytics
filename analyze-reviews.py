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