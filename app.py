from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

# CORS configuration
origins = ["*"]  # Allow all origins - for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for the request body
class PredictionRequest(BaseModel):
    total_tests: float
    population_density: float
    stringency_index: float
    hospital_beds_per_thousand: float
    gdp_per_capita: float

# Load the model
model_file_path = "model.joblib"
try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    # Train and save the model if it doesn't exist
    print("Model not found. Training and saving a new model...")

    # Create dummy data
    data = pd.DataFrame({
        'total_tests': np.random.rand(100),
        'population_density': np.random.rand(100),
        'stringency_index': np.random.rand(100),
        'hospital_beds_per_thousand': np.random.rand(100),
        'gdp_per_capita': np.random.rand(100),
        'new_cases': np.random.randint(0, 100, 100)
    })

    # Prepare the data
    X = data[['total_tests', 'population_density', 'stringency_index', 'hospital_beds_per_thousand', 'gdp_per_capita']]
    y = data['new_cases']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_file_path)
    print("Model training complete.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading or training model: {e}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Create a DataFrame from the request
        input_data = pd.DataFrame([request.dict()])

        # Scale the input data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)

        # Make the prediction
        prediction = model.predict(input_data)[0]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
