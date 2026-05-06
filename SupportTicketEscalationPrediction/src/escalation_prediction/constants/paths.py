"""
File paths and directories
"""
import os

#Directory Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
KAGGLE_CONFIG_DIR = "~/.kaggle"


# Kaggle specific Paths
KAGGLE_DATASET_DIR = "customer-support-tickets-crm-dataset"


# Output file names
COMBINED_DATA_OUTPUT = "raw_combined.pkl"
CLEANED_DATA_OUTPUT = "cleaned_data.pkl"

# EDA output directory
EDA_RESULTS_DIR = "results"

# EDA output filenames
EDA_CLASS_DISTRIBUTION_PLOT = "class_distribution.png"
EDA_TEXT_LENGTH_PLOT = "text_length_distribution.png"
EDA_SENTIMENT_PLOT = "sentiment_distribution.png"
EDA_WORDCLOUD_PLOT = "wordcloud.png"
EDA_TOP_WORDS_PLOT = "top_words.png"
EDA_SENTIMENT_ESCALATION_PLOT = "sentiment_escalation_plot.html"
EDA_DATA_SUMMARY = "data_summary.csv"

# Full file paths
RAW_COMBINED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, COMBINED_DATA_OUTPUT)
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, CLEANED_DATA_OUTPUT)
