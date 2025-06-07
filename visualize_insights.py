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

def plot_theme_distribution(df):
    """Plots the frequency of each theme, stacked by bank."""
    logging.info("Generating plot 3: Review Theme Distribution by Bank...")
    
    # Explode themes for accurate counting
    df_themes = df.copy()
    df_themes['theme'] = df_themes['theme'].str.split(', ')
    df_exploded = df_themes.explode('theme')
    
    # Create a crosstab for plotting
    theme_bank_crosstab = pd.crosstab(df_exploded['theme'], df_exploded['bank'])
    
    # Filter out 'General Feedback' for a cleaner plot if it dominates
    if 'General Feedback' in theme_bank_crosstab.index:
        theme_bank_crosstab = theme_bank_crosstab.drop('General Feedback')

    plt.figure()
    theme_bank_crosstab.plot(kind='barh', stacked=True, figsize=(12, 8))
    plt.title("Frequency of Review Themes by Bank")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Theme")
    plt.legend(title='Bank')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_theme_distribution_by_bank.png"))
    plt.close()

def generate_word_clouds(df):
    """Generates a word cloud for negative reviews for each bank."""
    logging.info("Generating plot 4: Word Clouds for Negative Reviews...")
    
    banks = df['bank'].unique()
    
    for bank in banks:
        plt.figure()
        # Filter for negative reviews for the current bank
        text = " ".join(review for review in df[(df['bank'] == bank) & (df['sentiment_label'] == 'NEGATIVE')]['review'])
        
        if not text:
            logging.warning(f"No negative reviews to generate word cloud for {bank}.")
            continue
            
        wordcloud = WordCloud(
            background_color="white",
            width=800,
            height=400,
            colormap='Reds',
            max_words=100,
            contour_width=3,
            contour_color='firebrick'
        ).generate(text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud of Pain Points for {bank}", fontsize=20)
        plt.savefig(os.path.join(OUTPUT_DIR, f"4_wordcloud_{bank.replace(' ', '_')}.png"))
        plt.close()

# --- Main Execution ---

def main():
    """Main function to orchestrate the visualization pipeline."""
    # 1. Load Data
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}. Please run Task 2 first.")
        return

    logging.info(f"Loading analyzed data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Setup and Generate Plots
    setup_plots()
    plot_rating_distribution(df)
    plot_sentiment_by_bank(df)
    plot_theme_distribution(df)
    generate_word_clouds(df)

    logging.info(f"All visualizations have been saved to the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()