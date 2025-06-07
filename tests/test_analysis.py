# tests/test_analysis.py
import pytest
# Import functions to be tested
from analyze_reviews import preprocess_text, assign_themes

def test_preprocess_text_lemmatization_and_stopwords():
    """Test that spaCy preprocessing correctly lemmatizes and removes stopwords."""
    text = "The app was crashing and running very slowly"
    expected_lemmas = ['app', 'crash', 'run', 'slowly']
    assert preprocess_text(text) == expected_lemmas

def test_assign_themes_single_theme():
    """Test assignment of a single, clear theme."""
    lemmas = ['app', 'crash', 'bug', 'fix']
    # 'crash' and 'bug' should map to 'Reliability & Bugs'
    assert assign_themes(lemmas) == 'Reliability & Bugs'

def test_assign_themes_multiple_themes():
    """Test assignment of multiple themes from different keywords."""
    lemmas = ['login', 'password', 'slow', 'transfer']
    # 'login' -> Account & Login Issues
    # 'slow', 'transfer' -> Transaction Performance
    assigned = assign_themes(lemmas)
    assert 'Account & Login Issues' in assigned
    assert 'Transaction Performance' in assigned

def test_assign_themes_no_match():
    """Test that it returns 'General Feedback' when no keywords match."""
    lemmas = ['this', 'is', 'a', 'great', 'thing']
    assert assign_themes(lemmas) == 'General Feedback'