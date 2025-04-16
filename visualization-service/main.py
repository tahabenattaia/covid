from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List, Dict, Optional

app = FastAPI(title="COVID Visualization Service",
             description="Service de visualisation des données COVID",
             version="1.0.0")

# Configuration de la base de données
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/covid_visualizations"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Visualization(Base):
    __tablename__ = "visualizations"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    image_data = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

class VisualizationInput(BaseModel):
    visualization_type: str
    country: Optional[str] = None
    features: Optional[List[str]] = None

class VisualizationOutput(BaseModel):
    image_data: str
    created_at: datetime

# Chargement des données
try:
    data = pd.read_csv('data/covid.csv')
except:
    data = None

@app.post("/visualize", response_model=VisualizationOutput)
async def create_visualization(input_data: VisualizationInput):
    try:
        if data is None:
            raise HTTPException(status_code=500, detail="Les données ne sont pas chargées")

        plt.figure(figsize=(10, 6))
        
        if input_data.visualization_type == "cases_by_country":
            if not input_data.country:
                raise HTTPException(status_code=400, detail="Le pays est requis pour ce type de visualisation")
            
            country_data = data[data['location'] == input_data.country]
            plt.plot(country_data['date'], country_data['total_cases'])
            plt.title(f'Évolution des cas de COVID-19 en {input_data.country}')
            plt.xlabel('Date')
            plt.ylabel('Nombre total de cas')
            plt.xticks(rotation=45)
            
        elif input_data.visualization_type == "correlation_matrix":
            features = input_data.features or [
                'total_cases', 'gdp_per_capita', 'hospital_beds_per_thousand',
                'life_expectancy', 'human_development_index'
            ]
            
            correlation_matrix = data[features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Matrice de corrélation')
            
        elif input_data.visualization_type == "feature_distribution":
            if not input_data.features:
                raise HTTPException(status_code=400, detail="Les features sont requises pour ce type de visualisation")
            
            for feature in input_data.features:
                sns.histplot(data[feature], kde=True, label=feature)
            plt.title('Distribution des features')
            plt.legend()
            
        else:
            raise HTTPException(status_code=400, detail="Type de visualisation non supporté")
        
        plt.tight_layout()
        
        # Conversion de l'image en base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return VisualizationOutput(
            image_data=img_base64,
            created_at=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": data is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 