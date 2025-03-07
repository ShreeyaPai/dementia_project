from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's URL for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Load Models
ann_model = joblib.load("models/ann_model.pkl")
label_encoder_gender = joblib.load("models/label_encoder_gender.pkl")  # Load fitted LabelEncoder
scaler = joblib.load("models/scaler.pkl")  # Load trained StandardScaler

# Define Input Schema
class PatientData(BaseModel):
    gender: str
    age: float
    EDUC: float
    SES: float
    MMSE: float
    CDR: float
    eTIV: float
    nWBV: float
    ASF: float

# Define Mapping
result_mapping = {0: "Non-Demented", 1: "Demented"}

@app.post("/predict/")
def predict(data: PatientData):
    # features = np.array([[data.gender, data.age, data.EDUC, data.SES, data.MMSE, data.CDR, data.eTIV, data.nWBV, data.ASF]])
    input_data = pd.DataFrame([[data.gender,data.age,data.EDUC,data.SES,data.MMSE,data.CDR,data.eTIV,data.nWBV,data.ASF]],
    columns = ['M/F','Age','EDUC','SES','MMSE','CDR','eTIV','NWBV','ASF'])

    # label_encoder_gender = LabelEncoder()
    # scaler = StandardScaler()
    input_data['M/F'] = label_encoder_gender.transform(input_data['M/F'])
    input_data = scaler.transform(input_data)
    # Make Predictions
    prediction = (ann_model.predict(input_data)>0.5).astype("int32")  # Returns an array
    # rfc_prediction = rfc_model.predict(features)  # Returns an array

    # Extract scalar values
    ann_pred_label = prediction.item()  # Convert array to scalar
    # rfc_pred_label = rfc_prediction.item()  # Convert array to scalar
    # print(prediction[0])
    return {
        "ANN_Prediction": result_mapping[ann_pred_label],
        # "RFC_Prediction": result_mapping[rfc_pred_label]
    }
