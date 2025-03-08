from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.getcwd(), "C:\\Users\\varsh\\Loan_approval_ML\\model\\loan_model.pkl")  # Ensure correct path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can specify frontend URL like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class LoanInput(BaseModel):
    features: list[float]  # Expecting a list of numerical values

@app.post("/predict")
def predict_loan(input_data: LoanInput):
    try:
        input_array = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
