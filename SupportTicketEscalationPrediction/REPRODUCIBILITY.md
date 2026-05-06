# Reproducibility Notes: Support Ticket Escalation Prediction

## What This Project Does

This project builds a machine learning pipeline that predicts whether a support ticket is likely to escalate before it actually does. It combines standard ML classifiers with a retrieval-augmented approach, meaning the system looks up historically similar resolved tickets at prediction time and uses what happened to them as additional evidence. The result is a measurable improvement in prediction accuracy compared to running the classifiers on their own.

---

## 1. What You Need to Run This

- Python 3.9 or higher
- Git (for cloning the repository)
- Any modern browser for the frontend dashboard — Chrome, Firefox, Safari, and Edge all work

---

## 2. Getting Set Up

### Step 1 - Clone the repository

```bash
git clone https://github.com/NileshB1/SupportTicketEscalationPrediction-Final.git
cd SupportTicketEscalationPrediction
```

### Step 2 — Create a virtual environment

It is worth doing this properly rather than installing everything globally. On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

The key libraries this pulls in are scikit-learn (≥1.0.0) for the ML models, XGBoost (≥1.7.0) for gradient boosting, pandas and numpy for data handling, MLflow (≥1.28.0) for experiment tracking, and FastAPI with uvicorn for the backend server.

### Step 4 - Download the Kaggle dataset (optional)

If you want to pull the original Kaggle dataset directly rather than using the copies already in the `datasets/` folder:

```bash
# First, put your kaggle.json credentials file in ~/.kaggle/
# Then run:
python -m src.escalation_prediction.data_loader
```

---

## 3. Preparing the Data

Run these three steps in order. Each one produces output that the next step depends on.

### Load the raw datasets

```bash
python -m src.escalation_prediction.data_loader
```

This merges all three source datasets and writes the combined output to `data/processed/cleaned_data.csv`.

### Clean and preprocess

```bash
python -m src.escalation_prediction.preprocessor
```

This handles text normalisation, missing value treatment, and any structural cleaning. The output updates `data/processed/cleaned_data.csv` in place.

### Run exploratory data analysis

```bash
python -m src.escalation_prediction.eda
```

This generates two outputs: a summary statistics table saved to `results/data_summary.csv`, and an interactive sentiment vs. escalation visualisation saved to `results/sentiment_escalation_plot.html`.

---

## 4. How Features Are Built

### TF-IDF text representation

The TF-IDF vectoriser is configured to process both unigrams and bigrams, with a vocabulary cap of 15,000 terms and a minimum document frequency of 2 (terms that only appear in a single ticket are dropped as noise). This produces a sparse matrix of shape `(n_samples, 15000)`. The configuration lives in `src/backend/services/inference.py` and `src/scripts/train_baseline.py`.

### Behavioural features

On top of the text representation, a set of handcrafted features captures signals that raw word counts miss:

- Message length
- Punctuation count
- Sentiment score
- Whether urgency keywords are present
- How customer sentiment has progressed across the conversation
- Historical escalation rate for similar cases

These produce a dense feature matrix alongside the sparse TF-IDF output.

### Retrieval-augmented features

This is the part that differentiates this system from a standard classifier. For each incoming ticket, cosine similarity is computed against all historical tickets in the index. The three most similar resolved tickets are retrieved, and binary flags recording whether each of those neighbours escalated are added as extra features. The idea is to ground predictions in concrete historical precedent rather than relying entirely on learned statistical patterns.

### Combined input

All three feature sources are stacked horizontally before training and inference:

```python
X = hstack([tfidf_matrix, engineered_features, rag_features])
```

---

## 5. Training the Models

### Run the training pipeline

```bash
python -m src.scripts.train_baseline
```

This single command does the following:

1. Loads the preprocessed data from `data/processed/cleaned_data.csv`
2. Fits the TF-IDF vectoriser on the training split
3. Extracts all engineered and retrieval-based features
4. Trains three classifiers: Logistic Regression, Random Forest (100 trees), and XGBoost (100 estimators, max depth 7)
5. Logs parameters and metrics to MLflow where available — if MLflow is not reachable it falls back gracefully and still saves results to disk
6. Serialises all trained models to `models/baseline/`

### Output files

| File | Contents |
|------|----------|
| `models/baseline/tfidf_vectorizer.pkl` | Fitted TF-IDF vectoriser |
| `models/baseline/logistic_regression.pkl` | Trained LR model |
| `models/baseline/random_forest.pkl` | Trained RF model |
| `models/baseline/xgboost.pkl` | Trained XGBoost model |
| `results/baseline_model_results.csv` | Per-model evaluation metrics |

### Checking results

```bash
cat results/baseline_model_results.csv
```

To browse the MLflow experiment UI:

```bash
mlflow ui
# Then open http://127.0.0.1:5000 in your browser
```

---

## 6. Running the Web Application

### Start the backend

```bash
uvicorn src.backend.main:app --reload --port 8000
```

Once running, the server is available at `http://127.0.0.1:8000`. The auto-generated API documentation (Swagger UI) is at `http://127.0.0.1:8000/docs` — useful if you want to test endpoints directly without going through the frontend.

### Open the dashboard

Navigate to:

```
http://127.0.0.1:8000/app/login.html
```

Demo credentials (hardcoded for the prototype):

- **Username:** `admin`
- **Password:** `admin123`

### Dashboard pages

| Page | Route | What it shows |
|------|-------|---------------|
| Login | `/app/login.html` | Authentication screen |
| Overview | `/app/predictive_analytics_overview.html` | High-risk ticket forecast and aggregate stats |
| Support Queue | `/app/support_queue.html` | Full ticket list with predicted risk levels |
| Ticket Detail | `/app/ticket_analysis_detail.html?id=<ticket_id>` | In-depth view: risk score, sentiment timeline, retrieved analogues |
| Model Performance | `/app/model_performance.html` | Side-by-side comparison of all three models |
| Performance Monitor | `/app/model_performance_monitor.html` | Live metrics and benchmark table |

### API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/login` | POST | Authenticate and receive JWT token |
| `/api/tickets` | GET | Fetch all tickets with predictions |
| `/api/tickets/{ticket_id}` | GET | Detailed analysis for a single ticket |
| `/api/dashboard_stats` | GET | Aggregate statistics for the overview screen |
| `/api/metrics` | GET | Model performance metrics |

---

## 7. Expected Results

### Baseline classifiers

Results vary slightly depending on the random seed used, but you should expect approximately:

- **Logistic Regression:** F1 ≈ 0.60, Precision ≈ 0.82
- **Random Forest:** F1 ≈ 0.72, Precision ≈ 0.91
- **XGBoost:** F1 ≈ 0.71, Precision ≈ 0.88

### With retrieval augmentation

- **F1 lift:** +15–18% over the best standard baseline
- **Precision:** 0.94 or above
- **Recall:** 0.89 or above

### Reproducing results with a fixed seed

```bash
# Remove any previous outputs (optional but recommended for a clean run)
rm -rf models/baseline/* mlflow.db

# Prepare the data
python -m src.escalation_prediction.data_loader
python -m src.escalation_prediction.preprocessor

# Set random seed before training (add np.random.seed(42) and random.seed(42)
# at the top of src/scripts/train_baseline.py if not already present)
python -m src.scripts.train_baseline

# Inspect the numbers
cat results/baseline_model_results.csv
```

---

```
SupportTicketEscalationPrediction/
    data/
      raw/                          
        - kaggle.json               # Kaggle API credentials
      processed/
        - cleaned_data.csv  
    datasets
       - customer_support_tickets.csv
       - rtweera_customer_care_emails_dataset.csv
       - synthetic-it-call-center-tickets_dataset.csv
    frontend/
      login.html     
      predictive_analytics_overview.html
      support_queue.html
      ticket_analysis_detail.html
      model_performance.html
      model_performance_monitor.html
   models/
      baseline/        
   results/
       baseline_model_results.csv  
       data_summary.csv                    # EDA results
       sentiment_escalation_plot.html
── src/
       __init__.py
       backend/
         api/                      # API route handlers
          services/
            - inference.py          # Model inference class
       escalation_prediction/
            __init__.py
           data_loader.py            # Load datasets
           preprocessor.py           #preprocessing
           eda.py                    #Exploratory data analysis
           baseline_models.py        # Baseline model definitions
          constants/
      
      scripts/
          __init__.py
           train_baseline.py         # Main training entry point
- mlruns/                           #
- README.md                         # Read Me file
- REPRODUCIBILITY.md                # reproducibility notes
- requirements.txt                  # Python dependencies
- Output.txt                        # 
```

---

## 9. Configuration Reference

### Text preprocessing

| Setting | Value |
|---------|-------|
| Lowercase | Yes |
| Remove punctuation | Yes |
| Remove stop words | Yes |
| Tokenisation | Whitespace-based |

### Feature engineering

| Component | Parameter | Value |
|-----------|-----------|-------|
| TF-IDF vocabulary size | max_features | 15,000 |
| TF-IDF n-gram range | ngram_range | (1, 2) |
| TF-IDF minimum document frequency | min_df | 2 |
| RAG neighbours retrieved | k | 3 |

### Model hyperparameters

| Model | Hyperparameters |
|-------|----------------|
| Logistic Regression | max_iter=5000, tol=1e-3, C=1.0 |
| Random Forest | n_estimators=100, max_depth=10, random_state=42 |
| XGBoost | n_estimators=100, max_depth=7, learning_rate=0.1 |

### Training split

- Train / Test: 80% / 20%
- Random seed: 42

---

## 10. Common Issues and Fixes

**"Module not found" errors**

This almost always means the virtual environment is not active. Run `source venv/bin/activate` (or `venv\Scripts\activate` on Windows), then try again.

**MLflow database schema error**

Delete the existing database file and let the training script create a fresh one:

```bash
rm mlflow.db
python -m src.scripts.train_baseline
```

**CORS errors in the browser**

Make sure the FastAPI backend is actually running on port 8000 before opening the frontend. CORS headers should already be configured in `src/backend/main.py` — if you are hosting the frontend on a different origin you may need to update the allowed origins list there.

**Data files not found**

You need to run the data loader and preprocessor before training or starting the server:

```bash
python -m src.escalation_prediction.data_loader
python -m src.escalation_prediction.preprocessor
```

**Slow inference on large queues**

A few things that help: reduce TF-IDF `max_features` from 15,000 to somewhere in the 5,000–10,000 range; cache the TF-IDF vectors so they are not recomputed on each request; drop the RAG k value to 1 or 2; or enable GPU acceleration for XGBoost if you have CUDA available.

---

## 11. Extending the System

### Adding a new dataset

1. Drop the CSV into the `datasets/` folder
2. Open `data_loader.py` and add the new source to the loading logic
3. Make sure the file has columns named `ticket_id`, `text`, and `escalated`
4. Re-run the data loader: `python -m src.escalation_prediction.data_loader`

### Experimenting with different hyperparameters

Open `src/scripts/train_baseline.py` and edit the model definitions directly. For example, to push XGBoost harder:

```python
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    random_state=42
)
```

Then re-run: `python -m src.scripts.train_baseline`

### Changing the feature engineering

Two methods in `src/backend/services/inference.py` control how features are built:

- `_extract_engineered_features()` — add or remove behavioural features here
- `_extract_rag_features()` — adjust the similarity threshold or the number of neighbours retrieved

---

## 12. Performance Benchmarks

### Inference speed (on standard laptop hardware)

| Task | Expected latency |
|------|-----------------|
| Single ticket prediction | 10–50 ms |
| Batch of 100 tickets | 500–1,500 ms |
| Dashboard load (20 tickets + stats) | 1–3 seconds |

### Saved model sizes

| Artefact | Approximate size |
|----------|-----------------|
| TF-IDF vectoriser | ~5 MB |
| Logistic Regression | ~2 MB |
| Random Forest | ~20 MB |
| XGBoost | ~15 MB |

---

## 13. Quick Validation Test

Once training is done, you can run a quick sanity check directly in Python:

```python
import joblib
from src.escalation_prediction.preprocessor import TextProcessor
from src.backend.services.inference import TicketInferenceAnalytics

# Load the fitted artefacts
tfidf = joblib.load('models/baseline/tfidf_vectorizer.pkl')
xgb   = joblib.load('models/baseline/xgboost.pkl')

# Try a sample ticket
sample_text = "Customer very upset with delayed response. Needs immediate assistance."
processor = TextProcessor()
cleaned  = processor.process(sample_text)

# Predict
tfidf_vec  = tfidf.transform([cleaned])
prediction = xgb.predict(tfidf_vec)
print(f"Escalation predicted: {prediction[0]}")
```

If everything is set up correctly, this should return a prediction of `1` for that example.

---

## 14. Reproducibility Checklist

Before recording or submitting, confirm each of these:

- [ ] Python 3.9+ installed and the virtual environment is active
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] All three dataset files are present in `datasets/`
- [ ] Data loader runs successfully and produces `cleaned_data.csv`
- [ ] Preprocessor runs successfully
- [ ] Training script completes and saves `.pkl` files to `models/baseline/`
- [ ] FastAPI server starts on port 8000 without errors
- [ ] Login page loads and the demo credentials work
- [ ] Tickets and risk predictions appear in the support queue
- [ ] Ticket detail page shows the sentiment timeline and retrieved analogues
- [ ] Model performance metrics are visible at `/api/metrics`

---

## 15. References

### Libraries used

- [scikit-learn](https://scikit-learn.org) — baseline classifiers and TF-IDF
- [XGBoost](https://xgboost.readthedocs.io) — gradient boosted trees
- [MLflow](https://mlflow.org) — experiment tracking and model registry
- [FastAPI](https://fastapi.tiangolo.com) — backend API framework

### Datasets

- Kaggle: Customer Support Tickets CRM Dataset (`ajverse/customer-support-tickets-crm-dataset`)
- HuggingFace: Customer Care Emails (`rtweera/customer_care_emails`)
- HuggingFace: Synthetic IT Call Centre Tickets (`KameronB/synthetic-it-callcenter-tickets`)

---

### Pointers
- Outputs from all steps are consolidated in `SupportTicketEscalationPrediction/Notes/final.txt`.
