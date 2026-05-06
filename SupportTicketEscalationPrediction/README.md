# Early Prediction of Customer Escalation in Support Tickets  
### (Context-Aware Retrieval + Machine Learning)

**Nilesh Barge (25168304)**  
MSc Data Analytics: National College of Ireland  
Module: Data Mining & Machine Learning (2025–2026)

---

## What this project is about  

In most support systems, escalation is expensive — not just in cost, but also in time and customer experience. Usually, tickets get escalated only after multiple back-and-forth interactions, which is already too late.

So the main idea here was:  
> *Can we predict escalation early, using both ML and past ticket context?*

Instead of relying only on structured features, I tried combining:
- traditional ML models  
- + similarity-based retrieval (RAG-style idea)

The assumption is that **tickets that “look like” previously escalated tickets are likely to escalate again**.

---

## What I actually built  

A hybrid pipeline that:
1. Takes a new ticket  
2. Finds similar historical tickets using TF-IDF similarity  
3. Extracts context from those tickets  
4. Combines that with engineered features  
5. Feeds everything into an ML classifier (mainly XGBoost)

So it's not a pure LLM RAG system — more like a **lightweight retrieval-augmented ML pipeline**.

---

## Research Questions  

- **RQ1:** Does adding retrieval improve performance over standard ML?  
- **RQ2:** Does sentiment progression across conversations help?  

---

## Project Structure  
```
SupportTicketEscalationPrediction/

src/
    backend/
        main.py                 : API entry point (FastAPI)
        services/
            inference.py          : Main prediction logic (RAG + ML)
    escalation_prediction/
        baseline_models.py      : ML code
        data_loader.py
        eda.py
        preprocessor.py
        constants/
    scripts/
        __init__.py
        train_baseline.py        : Script to train ML models

frontend/
    login.html
    predictive_analytics_overview.html
    support_queue.html
    ticket_analysis_detail.html
    model_performance_monitor.html

data/
    raw/
    processed/
        cleaned_data.pkl
        raw_combined.pkl
        cleaned_data.csv

models/
    baseline/
        XGBoost.pkl
        RandomForest.pkl
        LogisticRegression.pkl

results/
    baseline_model_results.csv
    confusion_matrix_*.png
    roc_curve_*.png
    precision_recall_curve_*.png
    Intermediate + Proposal PDFs

mlruns/                      : MLflow logs
other/                       : Personal notes
requirements.txt
README.md
```


---

## Dataset  

I merged 3 datasets (HuggingFace + Kaggle) and cleaned them into one.

Final dataset:

- **Total tickets:** 29,861  
- **Escalated:** ~53.6%  
- **Non-escalated:** ~46.4%  

### Features available:
- raw ticket text  
- processed text  
- sentiment score (-1 to +1)  
- escalation label  

Nothing too fancy, but enough to experiment.

---

## Feature Engineering  

This is where most of the effort went.

### Text features:
- TF-IDF (unigram + bigram, ~15k features)

### Behavioural features:
- sentiment score  
- text length  
- word count  
- avg word length  

### Heuristic signals:
- number of `!` → frustration  
- number of `?` → confusion  
- uppercase words → urgency  
- keyword flag (`urgent`, `blocked`, `critical`, etc.)

---

## Retrieval Component (Important Part)  

For each incoming ticket:
- compute TF-IDF vector  
- find top-K similar tickets  
- use those as additional context  

This is what gives the “RAG-like” behaviour.

Not using embeddings/LLMs here — just classic TF-IDF cosine similarity.

---

## Models Used  

Baseline:
- Logistic Regression  
- Random Forest  
- XGBoost  

Final model:
- XGBoost + retrieval features  

Train/test split: **80/20 (stratified)**

---

## Results  

| Model | F1 Score | ROC-AUC |
|------|---------|---------|
| **RAG + XGBoost** | **0.965** | **0.981** |
| XGBoost | 0.713 | 0.731 |
| Random Forest | 0.721 | 0.727 |
| Logistic Regression | 0.598 | 0.636 |

### Key takeaway:
Adding retrieval gave a **massive boost (~35% F1 improvement)**.

So context > just features.

---

## Web App  

Built a simple FastAPI app + HTML dashboard.

## How to run the project
Activate the environment and install requirements.txt
venv\Scripts\activate.bat
pip install -r requirements.txt

Run the pipeline from the project root in this order(from root folder "SupportTicketEscalationPrediction"):

1. Build the combined raw dataset

```powershell
python -m src.escalation_prediction.data_loader
```

2. Clean and preprocess the text

```powershell
python -m src.escalation_prediction.preprocessor
```

3. Generate plots, figures, and summary tables

```powershell
python -m src.escalation_prediction.eda
```

4. Train the baseline models and save the .pkl outputs
```powershell
The runnable script is `src/scripts/train_baseline.py`; the model implementation lives in `src/escalation_prediction/baseline_models.py`.
```

```powershell
python -m src.scripts.train_baseline
```

5. Start the FastAPI server

```powershell
uvicorn src.backend.main:app --port 8000
```

### Run it:

```bash
# Ensure you run this from the project root
#Example:
#PS D:\NCI\DataAnalytics\Projects\Data Mining And Machine Learning-11Feb26\code\SupportTicketEscalationPrediction> uvicorn src.backend.main:app --port 8000

#Output

Successfully trained live inference model to guarantee stable predictions.
Inference engine loaded successfully.
INFO:     Started server process [16076]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

---
Open:
http://127.0.0.1:8000 or http://localhost:8000

Login:
ID: admin
Password: admin123

---

## API endpoints

- `POST /api/login`
- `GET /api/tickets`
- `GET /api/tickets/{ticket_id}`
- `GET /api/metrics`
- `GET /api/health`

---

## Installation  

```powershell
venv\Scripts\activate.bat
pip install -r requirements.txt
```
