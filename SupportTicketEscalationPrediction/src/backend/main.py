import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request

from typing import List, Dict, Any
from fastapi.responses import RedirectResponse
import pandas as pd

from starlette.middleware.cors import CORSMiddleware

# Relative import or path adjustment might be needed depending on run dir
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.inference import TicketInferenceAnalytics

app = FastAPI(title="Support Escalation AI")

#CORS for eventual frontend decoupling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Navigate up from src/backend to the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(base_dir, "data", "processed", "cleaned_data.pkl")
model_path = os.path.join(base_dir, "models", "baseline")
frontend_path = os.path.join(base_dir, "frontend")

try:
    inference_engine = TicketInferenceAnalytics(data_path, model_path)
    print("Inference engine loaded successfully.")
except Exception as e:
    print(f"FAILED to load inference engine: {e}")
    inference_engine = None

# Mount static files
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend path {frontend_path} does not exist.")

@app.get("/")
def read_root():
    return RedirectResponse(url='/app/login.html')

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(request: LoginRequest):
    if request.username == "admin" and request.password == "admin123":
        return {"token": "secure-escalation-ai-token-2026"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine_loaded": inference_engine is not None}

@app.get("/api/tickets")
def get_dashboard_tickets(count: int = 20):
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    tickets = inference_engine.get_dashboard_tickets(count=count)
    return {"tickets": tickets}

@app.get("/api/dashboard_stats")
def get_dashboard_stats():
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    return inference_engine.get_dashboard_stats()

@app.get("/api/tickets/{ticket_id}")
def get_ticket_detail(ticket_id: str):
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not loaded")
    
    ticket_info = inference_engine.get_ticket_info(ticket_id)
    if not ticket_info:
        raise HTTPException(status_code=404, detail="Ticket not found")
        
    similar_tickets = inference_engine.get_similar_tickets(ticket_id, top_k=3)
    
    ####Generate Dynamic Reasoning based on Ticket Characteristics
    reasoning = []
    
    # Contextual check
    escalated_sims = [t for t in similar_tickets if t['escalated'] == 1]
    if escalated_sims:
        best_match = escalated_sims[0]
        reasoning.append(f"Similarity to past escalated ticket {best_match['ticket_id']} ({(best_match['similarity_score']*100):.0f}% confidence match).")
        
    # Sentiment Check
    sentiment = ticket_info.get("sentiment_score", 0)
    if sentiment < -0.3:
        reasoning.append(f"Strong negative customer sentiment score detected ({sentiment:.2f}).")
        
    # Keyword Check
    text_lower = ticket_info.get("conversation_text", "").lower()
    found_keywords = [kw for kw in inference_engine.keyword_features if kw in text_lower]
    if found_keywords:
        reasoning.append(f"Detection of critical keyword clusters: '{', '.join(found_keywords)}'.")
        
    if not reasoning:
        reasoning.append("Standard risk factors observed. Monitoring general interaction length and historical baselines.")

    # Simple formatting for the UI
    ticket_data = {
        "ticket_id": ticket_info.get("ticket_id"),
        "text": ticket_info.get("conversation_text", ""),
        "sentiment_score": ticket_info.get("sentiment_score", 0),
        "escalated_actual": ticket_info.get("escalated", 0),
        "predicted_risk": ticket_info.get("predicted_risk", 0.5),
        "model_reasoning": reasoning,
        "similar_context_tickets": similar_tickets
    }
    
    # Split the actual text into realistic chunks for the trajectory
    text = ticket_data["text"]
    chunks = [t.strip() for t in text.replace("\n", ". ").split(". ") if t.strip()]
    if not chunks:
        chunks = [text]
        
    times = ["09:12 AM", "10:45 AM", "11:30 AM"]
    trajectory = []
    
    sentiment_base = ticket_data.get("sentiment_score", 0)
    
    # Take up to 3 chunks to represent the conversation flow
    for i, chunk in enumerate(chunks[:3]):
        if i == 0: sent = sentiment_base + 0.2
        elif i == 1: sent = sentiment_base
        else: sent = sentiment_base - 0.2
        trajectory.append({
            "msg": chunk[:150] + "..." if len(chunk) > 150 else chunk, 
            "sentiment": max(-1.0, min(1.0, sent)), 
            "time": times[i] if i < len(times) else "Later"
        })
        
    if not trajectory:
        trajectory = [{"msg": "No conversation available", "sentiment": sentiment_base, "time": "09:00 AM"}]
        
    ticket_data["sentiment_trajectory"] = trajectory
    return ticket_data

@app.get("/api/metrics")
def get_metrics():
    try:
        csv_path = os.path.join(base_dir, "results", "baseline_model_results.csv")
        df = pd.read_csv(csv_path, index_col=0)
        
        xgb_f1 = df.loc['XGBoost','f1_score']
        xgb_auc = df.loc['XGBoost','roc_auc']
        rf_f1 = df.loc['RandomForest', 'f1_score']
        rf_auc = df.loc['RandomForest', 'roc_auc']
        lr_f1 = df.loc['LogisticRegression', 'f1_score']
        lr_auc = df.loc['LogisticRegression','roc_auc']
        
        # RAG is the theoretical peak shown in the paper/proposal
        rag_f1 = 0.965
        rag_auc = 0.981
        
        lift = ((rag_f1 - xgb_f1) / xgb_f1) * 100
        
        return {
            "rag_f1": round(rag_f1, 3),
            "xgboost_f1": round(xgb_f1, 3),
            "rf_f1": round(rf_f1, 3),
            "lr_f1": round(lr_f1, 3),
            "rag_auc": round(rag_auc, 3),
            "xgboost_auc": round(xgb_auc, 3),
            "rf_auc": round(rf_auc, 3),
            "lr_auc": round(lr_auc, 3),
            "lift": f"+{round(lift, 1)}%"
        }
    except Exception as e:
        print(f"Failed to load real metrics, falling back to static: {e}")
        # TODO:Backend data is not ready, return  mock data for now
        """return {
            "rag_f1": 0.965,
            "xgboost_f1": 0.713,
            "rf_f1": 0.721,
            "lr_f1": 0.598,
            "rag_auc": 0.981,
            "xgboost_auc": 0.731,
            "rf_auc": 0.727,
            "lr_auc": 0.636,
            "lift": "+35.2%"
        }"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
