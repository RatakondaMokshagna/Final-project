import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

app = FastAPI(title="Public Health Sentiment Dashboard")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Models
print("Loading NLP Models...")
try:
    sentiment_model = load_model("sentiment_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    misinfo_model = joblib.load("misinfo_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load models. Error: {e}")
    sentiment_model, tokenizer, label_encoder, misinfo_model, tfidf = None, None, None, None, None

MAX_LEN = 100

class TweetInput(BaseModel):
    text: str

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_text(data: TweetInput):
    text = data.text
    if not sentiment_model:
        return {"error": "Models not loaded"}
    
    # Sentiment prediction
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred_prob = sentiment_model.predict(padded)
    pred_class = pred_prob.argmax(axis=1)[0]
    sentiment = label_encoder.inverse_transform([pred_class])[0]
    
    # Misinfo prediction
    text_tfidf = tfidf.transform([text])
    misinfo_pred = misinfo_model.predict(text_tfidf)[0]
    is_misinfo = bool(misinfo_pred)
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": float(np.max(pred_prob)),
        "is_misinformation": is_misinfo
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
