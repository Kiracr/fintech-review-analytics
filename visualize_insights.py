# visualize_insights.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INPUT_CSV = "ethiopian_bank_reviews_analyzed.csv"
OUTPUT_DIR = "visuals" # Directory to save plots

# --- Plotting Functions ---

def setup_plots():
    """Set up consistent styling for all plots."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")

def plot_rating_distribution(df):
    """Plots the overall distribution of star ratings."""
    logging.info("Generating plot 1: Overall Star Rating Distribution...")
    plt.figure()
    ax = sns.countplot(x='rating', data=df, order=[1, 2, 3, 4, 5])
    ax.set_title("Overall Star Rating Distribution Across All Banks")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, "1_rating_distribution.png"))
    plt.close()

def plot_sentiment_by_bank(df):
    """Compares sentiment scores across the three banks using a box plot."""
    logging.info("Generating plot 2: Sentiment Score Distribution by Bank...")
    plt.figure()
    sns.boxplot(x='bank', y='sentiment_score', data=df)
    plt.title("Sentiment Score Distribution by Bank")
    plt.xlabel("Bank")
    plt.ylabel("Sentiment Score (-1: Negative, 1: Positive)")
    plt.xticks(rotation=10)
    plt.savefig(os.path.join(OUTPUT_DIR, "2_sentiment_by_bank.png"))
    plt.close()