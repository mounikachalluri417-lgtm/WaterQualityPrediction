
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Water Quality Prediction API")

model = joblib.load("water_quality_model.pkl")

class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {"message": "Water Quality Prediction API is running"}

@app.post("/predict")
def predict(sample: WaterSample):
    data = pd.DataFrame([sample.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "prediction": prediction,
        "result": "Potable" if prediction == 1 else "Not Potable",
        "note": "Educational prediction only; not a drinking-water safety certification."
    }
