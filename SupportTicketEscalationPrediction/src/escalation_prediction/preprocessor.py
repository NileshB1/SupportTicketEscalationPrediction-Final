import pandas as pd
import numpy as np
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import (
    STANDARD_TICKET_ID_COL,
    STANDARD_TEXT_COL,
    STANDARD_ESCALATED_COL,
    STANDARD_SENTIMENT_COL,
    STANDARD_CONVERSATION_TEXT_COL,
    STANDARD_PROCESSED_TEXT_COL,
    RAW_COMBINED_DATA_PATH,
    CLEANED_DATA_PATH,
)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

#Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        #adding stop words
        self.stop_words.update(['ticket', 'support', 'customer', 'please', 'thank', 'hi', 'hello', 'need'])

    def clean_text(self, text):
        """
        clean and normalize text data
        
        Args:
            text: The input text
        Returns:
            str: cleaned text       
        """
        if not isinstance(text, str):
            return ""
        text=text.lower()
        text=re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip() #remove white spaces
        return text
    
    def remove_stopwords_and_lemmatize(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def preprocess_text(self, text):
        """
        Full text preprocessing pipeline
        """
        text = self.clean_text(text)
        text = self.remove_stopwords_and_lemmatize(text)
        return text
    
def handle_missing_values(df):
    """
    handles missing values in the dataset
    """
    logger.info(f"Handling missing values")
    df[STANDARD_CONVERSATION_TEXT_COL] = df[STANDARD_CONVERSATION_TEXT_COL].fillna("")
    df[STANDARD_SENTIMENT_COL] = df[STANDARD_SENTIMENT_COL].fillna(0)
    df[STANDARD_ESCALATED_COL] = df[STANDARD_ESCALATED_COL].fillna(0)
    #drop rows where ticket_id is missing
    df = df.dropna(subset=[STANDARD_TICKET_ID_COL])
    logger.info(f"After handling missing values: {len(df)} rows")
    return df

def balance_datasets(X, y):
    """
    Balance the dataset using SMOTE.

    Args:
        X (pd.DataFrame or np.array): Features.
        y (pd.Series or np.array): Target.

    Returns:
        tuple: Balanced X and y.
    """
    logger.info(f"Applying SMOTE  for class balancing")
    smote =SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    logger.info(f"Balanced dataset: {len(y_balanced)} samples, {sum(y_balanced)} positive")
    return X_balanced, y_balanced

def pre_process_data(input_path, output_path):
    """
    Main preprocessing function
    """
    logger.info(f"Starting data preprocessing")
    df = pd.read_pickle(input_path)
    df = handle_missing_values(df)

    preprocessor = TextProcessor()
    df[STANDARD_PROCESSED_TEXT_COL] = df[STANDARD_CONVERSATION_TEXT_COL].apply(preprocessor.preprocess_text)

    df.to_pickle(output_path)
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    input_path = RAW_COMBINED_DATA_PATH
    output_path = CLEANED_DATA_PATH
    pre_process_data(input_path, output_path)


