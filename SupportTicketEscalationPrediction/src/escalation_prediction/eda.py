import os
import logging
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sbn
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
from .constants import (
    CLEANED_DATA_PATH,
    EDA_RESULTS_DIR,
    EDA_CLASS_DISTRIBUTION_PLOT,
    EDA_TEXT_LENGTH_PLOT,
    EDA_SENTIMENT_PLOT,
    EDA_WORDCLOUD_PLOT,
    EDA_TOP_WORDS_PLOT,
    EDA_SENTIMENT_ESCALATION_PLOT,
    EDA_DATA_SUMMARY,
)

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_class_distribution(df, save_path):
    """
    Plot distribution of escalated versus non-escalated tickets
    """
    plot.figure(figsize=(8,6))
    sbn.countplot(x='escalated', data=df)
    plot.title('Distribution of Escalated vs Non-Escalated Tickets')
    plot.xlabel('Escalated (0=No, 1=Yes)')
    plot.ylabel('Count')
    plot.savefig(save_path)
    plot.close()
    logger.info(f"Saved class distribution plot to {save_path}")

def plot_text_length_distribution(df, save_path):
    """
    Plot the distribution of text lengths.

    Args:
        df (pd.DataFrame): The dataset
        save_path (str): Path to save the plot
    """
    df["text_length"] = df['conversation_text'].str.len()
    plot.figure(figsize=(10,6))
    sbn.histplot(df["text_length"], bins=50, kde=True)
    plot.title('Distribution of Conversation Text Lengths')
    plot.xlabel('Text Length (characters)')
    plot.ylabel('Frequency')
    plot.savefig(save_path)
    plot.close()
    logger.info(f"Saved text length distribution plot to {save_path}")

def plot_sentiment_distribution(df, save_path):
    """
    Plot the distribution of sentiment scores
    """
    plot.figure(figsize=(8, 6))
    sbn.histplot(df['sentiment_score'], bins=20, kde=True)
    plot.title('Distribution of Sentiment Scores')
    plot.xlabel('Sentiment Score')
    plot.ylabel('Frequency')
    plot.savefig(save_path)
    plot.close()
    logger.info(f"Saved sentiment distribution plot to {save_path}")

def generate_wordcloud(df, save_path):
    """
    Generate word cloud from conversation text
    """
    text = ' '.join(df['conversation_text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plot.figure(figsize=(10, 5))
    plot.imshow(wordcloud, interpolation='bilinear')
    plot.axis("off")
    plot.title('Word Cloud of Conversation Texts')
    plot.savefig(save_path)
    plot.close()
    logger.info(f"Saved word cloud to {save_path}")

def plot_top_words(df, save_path, top_n=20):
    """
    Plot the top N most frequent words.

    Args:
        df (pd.DataFrame): The dataset
        save_path (str): Path to save the plot
        top_n (int): Number of top words to plot
    """
    all_words = ' '.join(df["conversation_text"].dropna()).split()
    word_Counts = Counter(all_words)
    top_words = word_Counts.most_common(top_n)
    words, counts = zip(*top_words)

    plot.figure(figsize=(12, 8))
    sbn.barplot(x=list(counts), y=list(words))
    plot.title(f'Top {top_n} Most Frequent Words')
    plot.xlabel("Frequency")
    plot.ylabel("Words")
    plot.savefig(save_path)
    plot.close()
    logger.info(f"Saved top words plot to {save_path}")

def generate_interactive_sentiment_plot(df, save_path):
    """
    Generate an interactive plot for sentiment vs escalation.
    """
    fig = px.scatter(df, x='sentiment_score', y='escalated', color='escalated',
                     title='Sentiment Score vs Escalation',
                     labels={'sentiment_score': 'Sentiment Score', 'escalated': 'Escalated (0=No, 1=Yes)'},
                     hover_data=['conversation_text'])
    fig.write_html(save_path)
    logger.info(f"Saved interactive sentiment plot to {save_path}")

def perform_eda(data_path, results_path):
    """
    Perform exploratory data analysis and save plots.

    Args:
        data_path (str): Path to the processed data pickle.
        results_path (str): Path to the results directory.
    """
    logger.info("Starting EDA")
    df = pd.read_pickle(data_path)
    os.makedirs(results_path, exist_ok=True)
    plot_class_distribution(df, os.path.join(results_path, EDA_CLASS_DISTRIBUTION_PLOT))
    plot_text_length_distribution(df, os.path.join(results_path, EDA_TEXT_LENGTH_PLOT))
    plot_sentiment_distribution(df, os.path.join(results_path, EDA_SENTIMENT_PLOT))   
    generate_wordcloud(df, os.path.join(results_path, EDA_WORDCLOUD_PLOT))
    plot_top_words(df, os.path.join(results_path, EDA_TOP_WORDS_PLOT))
    generate_interactive_sentiment_plot(df, os.path.join(results_path, EDA_SENTIMENT_ESCALATION_PLOT))

    #generate summary statistics
    summary = df.describe(include='all')
    summary.to_csv(os.path.join(results_path, EDA_DATA_SUMMARY))
    logger.info(f"Saved data summary to {os.path.join(results_path, 'data_summary.csv')}")

    logger.info("EDA completed")

if __name__ == "__main__":
    data_path = CLEANED_DATA_PATH
    result_path = EDA_RESULTS_DIR
    perform_eda(data_path, result_path)