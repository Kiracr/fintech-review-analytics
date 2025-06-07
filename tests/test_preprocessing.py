# tests/test_preprocessing.py
import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import pytest
from datetime import datetime

# Import the function to be tested
from scrape_preprocess import preprocess_data, FINAL_COLUMNS

@pytest.fixture
def sample_raw_data():
    """Creates a sample raw DataFrame for testing."""
    data = {
        'reviewId': ['gp:1', 'gp:2', 'gp:3', 'gp:4', 'gp:5', 'gp:6'],
        'content': [
            'This is a good app!',        # Normal case
            None,                         # Missing content
            'This is a duplicate.',       # Duplicate
            'This is a duplicate.',       # Duplicate
            '  Needs more features. ',    # Has whitespace
            ''                            # Empty string
        ],
        'score': [5, 1, 3, 3, 4, 2],
        'at': [
            datetime(2023, 10, 26),
            datetime(2023, 10, 25),
            datetime(2023, 10, 24),
            datetime(2023, 10, 24),
            datetime(2023, 10, 23),
            datetime(2023, 10, 22),
        ],
        'bank': ['CBE', 'BOA', 'Dashen', 'Dashen', 'CBE', 'BOA'],
        'source': ['Google Play Store'] * 6,
    }
    return pd.DataFrame(data)

def test_preprocess_data_removes_nulls_and_empty(sample_raw_data):
    """Test that rows with null or empty 'content' are removed."""
    clean_df, _, _ = preprocess_data(sample_raw_data)
    # Expected to keep rows 0, 2, and 4 (after deduplication)
    assert len(clean_df) == 3
    assert 'This is a good app!' in clean_df['review'].values
    assert 'Needs more features.' in clean_df['review'].values

def test_preprocess_data_removes_duplicates(sample_raw_data):
    """Test that duplicate reviews based on reviewId are removed."""
    clean_df, _, _ = preprocess_data(sample_raw_data)
    # Check that 'This is a duplicate.' appears only once
    assert clean_df[clean_df['review'] == 'This is a duplicate.'].shape[0] == 1

def test_preprocess_data_normalizes_dates(sample_raw_data):
    """Test that the date format is correctly normalized to YYYY-MM-DD."""
    clean_df, _, _ = preprocess_data(sample_raw_data)
    # Check the format of the date column
    assert clean_df['date'].iloc[0] == '2023-10-26'
    assert all(clean_df['date'].str.match(r'\d{4}-\d{2}-\d{2}'))

def test_preprocess_data_column_structure(sample_raw_data):
    """Test that the final DataFrame has the correct columns and order."""
    clean_df, _, _ = preprocess_data(sample_raw_data)
    assert list(clean_df.columns) == FINAL_COLUMNS

def test_preprocess_data_strips_whitespace(sample_raw_data):
    """Test that leading/trailing whitespace is stripped from reviews."""
    clean_df, _, _ = preprocess_data(sample_raw_data)
    review_with_whitespace = clean_df[clean_df['rating'] == 4]['review'].iloc[0]
    assert review_with_whitespace == 'Needs more features.'