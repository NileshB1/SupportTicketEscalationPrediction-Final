import os
import logging
import pandas as pd
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi

from .constants import (
    # Dataset names
    HUGGINGFACE_CUSTOMER_CARE_EMAILS,
    HUGGINGFACE_SYNTHETIC_IT_TICKETS,
    KAGGLE_CUSTOMER_SUPPORT_TICKETS,
    DATASET_DICT,
    # Column names
    CCE_SOURCE_TEXT_COL,
    CCE_SOURCE_TICKET_COL,
    CCE_SOURCE_ESCALATION_COL,
    CCE_SOURCE_SENTIMENT_COL,
    CCE_ESCALATION_HIGH,
    CST_SOURCE_TICKET_COL,
    CST_SOURCE_TEXT_COL,
    CST_SOURCE_SUBJECT_COL,
    CST_SOURCE_PRIORITY_COL,
    CST_SOURCE_SATISFACTION_COL,
    CST_PRIORITY_HIGH,
    CST_PRIORITY_MEDIUM,
    SIT_SOURCE_TICKET_COL,
    SIT_SOURCE_TEXT_COL,
    SIT_SOURCE_REASSIGNED_COL,
    SIT_SOURCE_SENTIMENT_COL,
    STANDARD_TICKET_ID_COL,
    STANDARD_TEXT_COL,
    STANDARD_ESCALATED_COL,
    STANDARD_SENTIMENT_COL,
    # Paths
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    KAGGLE_CONFIG_DIR,
    KAGGLE_DATASET_DIR,
    COMBINED_DATA_OUTPUT,
    # Processing
    SATISFACTION_NORMALIZATION_FACTOR,
    DEFAULT_SATISFACTION_SCORE,
    DEFAULT_SATISFACTION_NORMALIZED,
    SATISFACTION_ESCALATION_THRESHOLD,
)

# logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, dataset_dict, output_path, kaggle_config_dir):
        """
        Initializes the dataset and other configurations
        """
        self.dataset_dict = dataset_dict
        self.output_path = output_path
        self.kaggle_config_dir = kaggle_config_dir

    def download_kaggle_datasets(self, dataset_name, output_path, kaggle_config_dir=None):
        """
        Downloads datasets from Kaggle with the help of Kaggle APIs
        Args:
            dataset_name: Kaggle dataset name ('owner/dataset-name')
            output_path: The local path where to save the dataset.
            kaggle_config_dir: Path to the directory containing kaggle.json

        Raises:
            Exception: if download fails
        """
        try:
            api = KaggleApi(config_dir=kaggle_config_dir) if kaggle_config_dir else KaggleApi()
            api.authenticate() # Authenticate using provided config
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            api.dataset_download_files(dataset_name, path=output_path, unzip=True)
            logger.info(f"Downloaded {dataset_name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to download from Kaggle dataset {dataset_name}, Error {e}")
            raise

    def load_huggingface_dataset(self, dataset_name):
        """
        Load a dataset from Hugging Face

        Args:
            dataset_name (str): The Hugging Face dataset name

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame
        """
        try:
            logger.info(f"Loading Huggingface dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)

            if hasattr(dataset, 'keys'):
                split_name = 'train' if 'train' in dataset.keys() else next(iter(dataset.keys()))
                df = pd.DataFrame(dataset[split_name])
            else:
                df = pd.DataFrame(dataset)

            logger.info(f"#### Huggingface dataset: {df.head()}")
            logger.info(f"Loaded {len(df)} rows from {dataset_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {str(e)}")
            raise

    def standardize_customer_care_emails(self, df):
        #rtweera/customer_care_emails
        ## Datasetlink1: https://huggingface.co/datasets/rtweera/customer_care_emails
        """
        Standardize the rtweera/customer_care_emails dataset

        Args:
            df: raw dataset

        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Mapping message_body as conversation_text, thread_id as ticket_id
        # escalated: 1 if email_criticality == 'high', else 0
        # sentiment_score: customer_satisfaction
        df = df.rename(columns={
            CCE_SOURCE_TEXT_COL: STANDARD_TEXT_COL,
            CCE_SOURCE_TICKET_COL: STANDARD_TICKET_ID_COL
        })
        df[STANDARD_ESCALATED_COL] = df[CCE_SOURCE_ESCALATION_COL].apply(
            lambda x: 1 if x == CCE_ESCALATION_HIGH else 0
        )
        df[STANDARD_SENTIMENT_COL] = df[CCE_SOURCE_SENTIMENT_COL]
        return df[[
            STANDARD_TICKET_ID_COL,
            STANDARD_TEXT_COL,
            STANDARD_SENTIMENT_COL,
            STANDARD_ESCALATED_COL
        ]]
    
    def standardize_customer_support_tickets(self, df):
        #ajverse/customer-support-tickets-crm-dataset
        #Dataset link2: https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset 
        """
        Standardize the ajverse/customer-support-tickets-crm-dataset.

        Args:
            df (pd.DataFrame): raw dataset

        Returns:
            pd.DataFrame: Standardized DataFrame.
        """
        # Map columns: Ticket_ID as ticket_id, Ticket_Description as conversation_text
        df = df.rename(columns={
            CST_SOURCE_TICKET_COL: STANDARD_TICKET_ID_COL,
            CST_SOURCE_TEXT_COL: STANDARD_TEXT_COL
        })
        
        # If Ticket_Description is missing, combine Ticket_Subject and Ticket_Description
        if STANDARD_TEXT_COL not in df.columns and CST_SOURCE_SUBJECT_COL in df.columns:
            df[STANDARD_TEXT_COL] = (
                df[CST_SOURCE_SUBJECT_COL].fillna('') + ' ' +
                df.get(CST_SOURCE_TEXT_COL, '').fillna('')
            )
        
        # Map escalation: 1 if (Priority_Level is High or Medium) AND (Satisfaction_Score is 1 or 2)
        def determine_escalation(row):
            try:
                priority = str(row.get(CST_SOURCE_PRIORITY_COL, '')).lower()
                satisfaction = int(row.get(CST_SOURCE_SATISFACTION_COL, DEFAULT_SATISFACTION_SCORE))
                return 1 if (
                    priority in [CST_PRIORITY_HIGH, CST_PRIORITY_MEDIUM]
                ) and (
                    satisfaction in SATISFACTION_ESCALATION_THRESHOLD
                ) else 0
            except (ValueError, TypeError):
                return 0
        
        df[STANDARD_ESCALATED_COL] = df.apply(determine_escalation, axis=1)
        
        # Use 'Satisfaction_Score' as sentiment_score (normalize to 0-1 if it's on 1-5 scale)
        df[STANDARD_SENTIMENT_COL] = df.get(
            CST_SOURCE_SATISFACTION_COL,
            DEFAULT_SATISFACTION_SCORE
        ).apply(
            lambda x: (float(x) - 1) / SATISFACTION_NORMALIZATION_FACTOR
            if pd.notna(x) else DEFAULT_SATISFACTION_NORMALIZED
        )
        
        return df[[
            STANDARD_TICKET_ID_COL,
            STANDARD_TEXT_COL,
            STANDARD_SENTIMENT_COL,
            STANDARD_ESCALATED_COL
        ]]

        
    def standardize_synthetic_it_tickets(self, df):
        #KameronB/synthetic-it-callcenter-tickets dataset
        # Dataset link3: https://huggingface.co/datasets/KameronB/synthetic-it-callcenter-tickets
        """
        Standardize the KameronB/synthetic-it-callcenter-tickets dataset.

        Args:
            df (pd.DataFrame): The raw dataset.

        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Map columns: number as ticket_id, content as conversation_text
        # escalated: 1 if reassigned_count > 0, else 0
        # sentiment_score: info_score_close_notes
        df = df.rename(columns={
            SIT_SOURCE_TICKET_COL: STANDARD_TICKET_ID_COL,
            SIT_SOURCE_TEXT_COL: STANDARD_TEXT_COL
        })
        df[STANDARD_ESCALATED_COL] = df[SIT_SOURCE_REASSIGNED_COL].apply(lambda x: 1 if x > 0 else 0)
        df[STANDARD_SENTIMENT_COL] = df[SIT_SOURCE_SENTIMENT_COL]
        return df[[
            STANDARD_TICKET_ID_COL,
            STANDARD_TEXT_COL,
            STANDARD_SENTIMENT_COL,
            STANDARD_ESCALATED_COL
        ]]
    
    def load_all_datasets(self, raw_data_path):
        """
        Load and standardize all datasets.

        Args:
            raw_data_path (str): Path to the raw data directory

        Returns:
            pd.DataFrame: Combined and standardized DataFrame
        """
        all_data = []

        # Load Hugging Face datasets
        try:
            df1 = self.load_huggingface_dataset(HUGGINGFACE_CUSTOMER_CARE_EMAILS)
            df1 = self.standardize_customer_care_emails(df1)
            all_data.append(df1)
        except Exception as e:
            logger.warning(f"Skipping {HUGGINGFACE_CUSTOMER_CARE_EMAILS}: {str(e)}")

        try:
            df3 = self.load_huggingface_dataset(HUGGINGFACE_SYNTHETIC_IT_TICKETS)
            df3 = self.standardize_synthetic_it_tickets(df3)
            all_data.append(df3)
        except Exception as e:
            logger.warning(f"Skipping {HUGGINGFACE_SYNTHETIC_IT_TICKETS}: {str(e)}")

        # Load Kaggle dataset
        kaggle_dataset_path = os.path.join(raw_data_path, KAGGLE_DATASET_DIR)
        kaggle_config_dir = raw_data_path  # Currently kaggle.json is in data/raw
        try:
            self.download_kaggle_datasets(KAGGLE_CUSTOMER_SUPPORT_TICKETS, raw_data_path, kaggle_config_dir)
            # downloaded file is a CSV
            csv_files = [f for f in os.listdir(kaggle_dataset_path) if f.endswith('.csv')]
            if csv_files:
                df2 = pd.read_csv(os.path.join(kaggle_dataset_path, csv_files[0]))
                df2 = self.standardize_customer_support_tickets(df2)
                all_data.append(df2)
            else:
                logger.warning("No CSV file found in Kaggle dataset")
        except Exception as e:
            logger.warning(f"Skipping {KAGGLE_CUSTOMER_SUPPORT_TICKETS}: {str(e)}")

        if not all_data:
            raise ValueError("No datasets could be loaded")

        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"#### Combined dataset has {len(combined_df)} rows")
        return combined_df

if __name__ == "__main__":
    data_loader = DataLoader(
        dataset_dict=DATASET_DICT,
        output_path=RAW_DATA_PATH,
        kaggle_config_dir=KAGGLE_CONFIG_DIR
    )

    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df = data_loader.load_all_datasets(RAW_DATA_PATH)
    # Save raw combined data
    output_file = os.path.join(PROCESSED_DATA_PATH, COMBINED_DATA_OUTPUT)
    df.to_pickle(output_file)
    logger.info(f"Saved raw combined data to {output_file}")