"""
Data processing constants
"""

TRAIN_SPLIT = "train"

SATISFACTION_SCALE_MIN = 1
SATISFACTION_SCALE_MAX = 5
SATISFACTION_NORMALIZATION_FACTOR = 4  # (max - min)
DEFAULT_SATISFACTION_SCORE = 3
DEFAULT_SATISFACTION_NORMALIZED = 0.5

SATISFACTION_ESCALATION_THRESHOLD = [1, 2]  # Scores indicating escalation

RAW_COMBINED_DATA_PATH = "data/processed/raw_combined.pkl"
CLEANED_DATA_PATH = "data/processed/cleaned_tickets.pkl"
