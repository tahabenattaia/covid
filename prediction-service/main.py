from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict
import joblib
import os
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="COVID Prediction Service",
             description="Service de prédiction des cas COVID",
             version="1.0.0")

class PredictionInput(BaseModel):
    gdp_per_capita: float
    hospital_beds_per_thousand: float
    life_expectancy: float
    human_development_index: float

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float

# Chemins des fichiers du modèle
MODEL_PATH = 'models/covid_model.joblib'
SCALER_PATH = 'models/scaler.joblib'

# Chargement du modèle et du scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except:
    model = None
    scaler = None

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé")
        
        # Préparation des features
        features = np.array([
            input_data.gdp_per_capita,
            input_data.hospital_beds_per_thousand,
            input_data.life_expectancy,
            input_data.human_development_index
        ]).reshape(1, -1)
        
        # Normalisation des features
        features_scaled = scaler.transform(features)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        
        # Calcul de la confiance (basé sur l'erreur moyenne du modèle)
        confidence = 0.85  # À ajuster selon les métriques réelles du modèle
        
        return PredictionOutput(
            prediction=float(prediction),
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 