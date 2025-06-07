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
