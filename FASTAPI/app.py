# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# app = FastAPI()

# model = joblib.load('model.pkl')
# class Flower (BaseModel):
#     sepal_length:float
#     sepal_width:float
#     petal_length:float
#     petal_width:float   
# @app.get("/")
# def home():
#     return{"message": "Welcome"}
# @app.post("/predict")
# def predict(data: Flower):
#     prediction=model.predict([[data.sepal_length, data.sepal_width,data.petal_length,data.petal_width]])
#     return{"prediction": int(prediction[0])}
    

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained Random Forest model
model = joblib.load("random_forest_ev_model.pkl")

# Define request schema
class EVFeatures(BaseModel):
    Electric_Range: float
    Model_Year: int
    Make: int
    Model: int
    County: int
    City: int

# Root endpoint
@app.get("/")
def home():
    return {"message": "EV Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(features: EVFeatures):
    data = np.array([[
        features.Electric_Range,
        features.Model_Year,
        features.Make,
        features.Model,
        features.County,
        features.City
    ]])
    
    prediction = model.predict(data)
    result = "BEV" if prediction[0] == 0 else "PHEV"
    
    return {"prediction": result}
