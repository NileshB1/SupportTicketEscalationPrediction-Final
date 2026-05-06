import os
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plot
import seaborn as sbn
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

import mlflow
import mlflow.sklearn

# logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    def __init__(self, data_path, model_path, results_path):
        """
        Initializes the paths and configutions
        """
        self.data_path = data_path
        self.model_path = model_path
        self.results_path = results_path
        self.models = {
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                solver='saga',
                n_jobs=-1
            ),
            'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        }
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.keyword_features = [
            'escalat', 'urgent', 'critical', 'unable', 'failure', 'blocked',
            'immediately', 'outage', 'incident', 'problem'
        ]
        self.X_train = None
        self.X_Test = None
        self.y_train = None
        self.y_test = None

    def _keyword_indicator(self, text: str) -> int:
        lower = str(text).lower()
        return int(any(keyword in lower for keyword in self.keyword_features))

    def _extract_engineered_features(self, df: pd.DataFrame):
        raw_text = df['conversation_text'].fillna('').astype(str)
        text_length = raw_text.str.len().values.reshape(-1, 1)
        word_count = raw_text.str.split().apply(len).values.reshape(-1, 1)
        avg_word_length = np.where(word_count.flatten() > 0,
                                   text_length.flatten() / word_count.flatten(),
                                   0).reshape(-1, 1)
        exclamation_count = raw_text.str.count(r'!').values.reshape(-1, 1)
        question_count = raw_text.str.count(r'\?').values.reshape(-1, 1)
        uppercase_count = raw_text.str.findall(r'[A-Z]{2,}').str.len().fillna(0).astype(int).values.reshape(-1, 1)
        keyword_flag = raw_text.apply(self._keyword_indicator).values.reshape(-1, 1)
        sentiment = df['sentiment_score'].fillna(0).values.reshape(-1, 1) if 'sentiment_score' in df.columns else np.zeros((len(df), 1))

        feature_array = np.hstack([
            text_length,
            word_count,
            avg_word_length,
            exclamation_count,
            question_count,
            uppercase_count,
            keyword_flag,
            sentiment
        ])
        return csr_matrix(feature_array)

    def train_and_eval_model(self, model_name: str, model) -> Dict[str, Any]:
        """
        Train and evaluate single model
        """ 
        logger.info(f"Training model is: {model_name}")
        model.fit(self.X_train, self.y_train)

        #pred
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:,1]

        # metrics
        report = classification_report(self.y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        #cv score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')

        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'roc_auc': roc_auc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
        }
        logger.info(f"Model {model_name}, F1 {metrics['f1_score']:.4f}, "
                    f"ROC-AUC: {metrics['roc_auc']:.4f})")
        return model, metrics 
    
    def load_data(self):
        """
        Loads the data from the specified path
        """
        logger.info(f"Loading data from path: {self.data_path}")
        df = pd.read_pickle(self.data_path)
        column_text = "processed_text" if "processed_text" in df.columns else "conversation_text"
        text_features = self.vectorizer.fit_transform(df[column_text].fillna(""))
        engineered_features = self._extract_engineered_features(df)
        self.y = df['escalated'].values
        self.X = hstack([text_features, engineered_features])
        class_counts = pd.Series(self.y).value_counts().to_dict()
        logger.info(
            f"Loaded {self.X.shape[0]} samples, total features {self.X.shape[1]}. "
            f"Class counts: {class_counts}"
        )


    def split_data(self, test_size=0.2):
        """
        Splits data into training set and test set(80-20 ratio)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        logger.info(f"Data split is: train {self.X_train.shape[0]}, test {self.X_test.shape[0]}")

    def plot_roc_curve(self, y_true, y_pred_probability, model_name):
        """
        Plot and save the ROC curve
        """
        false_pos_rate, true_pos_rate, _ = roc_curve(y_true, y_pred_probability)
        plot.figure(figsize=(8,6))
        plot.plot(false_pos_rate, true_pos_rate, 
                  label=f"ROC curve (AUC = {roc_auc_score(y_true, y_pred_probability):.4f})")
        plot.plot([0, 1], [0, 1], 'k--')
        plot.xlabel("False Positive Rate")
        plot.ylabel('True Positive Rate')
        plot.title(f"ROC Curve: {model_name}")
        plot.legend()
        plot.savefig(os.path.join(self.results_path, f"roc_curve_{model_name}.png"))
        plot.close()

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plots the confusion matrix
         Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: model name
        """
        confusion_metrix = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix for {model_name}: is \n{confusion_metrix}")
        plot.figure(figsize=(8,6))
        sbn.heatmap(confusion_metrix, annot=True, fmt='d', cmap='Blues')
        plot.title(f'Confusion Matrix: {model_name}')
        plot.xlabel("Predicted")
        plot.ylabel("Actual")
        plot.savefig(os.path.join(self.results_path, f"confusion_matrix_{model_name}.png"))
        plot.close()


    def save_model(self, model, model_name):
        """
        Saves the trained model.
        """
        model_file_path = os.path.join(self.model_path, f"{model_name}.pkl")
        joblib.dump(model, model_file_path)
        logger.info(f"Model {model_name} saved successfully, path: {model_file_path}")



    def plot_precision_recall_curve(self, y_true, y_pred_probability, model_name: str):
        """
        Plots the precision-recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probability)
        plot.figure(figsize=(8,6))
        plot.plot(recall, precision, label=f"Precision-Recall curve")
        plot.xlabel("Recall")
        plot.ylabel('Precision')
        plot.title(f"Precision-Recall Curve: {model_name}")
        plot.legend()
        plot.savefig(os.path.join(self.results_path, f"precision_recall_curve_{model_name}.png"))
        plot.close()

    def baseline_training_run(self):
        """
        Runs the baseline training pipeline
        """
        #setup mflow
        mlflow.set_experiment("Customer Escalation Prediction - Baseline Models")
        self.load_data()    #load data
        self.split_data()   # split data

        final_results = {}
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                trained_model, metrics = self.train_and_eval_model(model_name, model)
                #log metrics to mflow
                for metric_key, metric_value in metrics.items():
                    mlflow.log_metric(metric_key, metric_value)
                # log model
                mlflow.sklearn.log_model(trained_model, model_name)

                #Generate and save plot
                y_pred = trained_model.predict(self.X_test)
                y_pred_probability = trained_model.predict_proba(self.X_test)[:,1]

                self.plot_confusion_matrix(self.y_test, y_pred, model_name)
                self.plot_roc_curve(self.y_test, y_pred_probability, model_name)
                self.plot_precision_recall_curve(self.y_test, y_pred_probability, model_name)

                #Save the model locally
                self.save_model(trained_model, model_name)
                final_results[model_name]=metrics
        #save result summary
        results_df = pd.DataFrame(final_results).T
        results_df.to_csv(os.path.join(self.results_path, "baseline_model_results.csv"))
        logger.info(f"Baseline model completed.")

        return final_results
    

if __name__ == "__main__":
    data_path = "data/processed/cleaned_data.pkl"
    model_path = "models/baseline"
    results_path = "results"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    model_trainer = BaselineModelTrainer(data_path, model_path, results_path) #Class instance
    model_trainer.baseline_training_run()