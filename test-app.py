from fastapi import FastAPI, Path, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("model.joblib")

# Load data from a Parquet file
df = pd.read_csv("application_train.csv")

# Endpoint to make predictions with SK_ID_CURR as a path parameter in the URL
@app.get("/predict/{sk_id}")
def predict(sk_id: int = Path(..., title="SK_ID_CURR", description="The unique identifier of the client")):
    # Validate user input (e.g., check if SK_ID_CURR exists in the database)
    if not check_sk_id_exists(sk_id):
        raise HTTPException(status_code=404, detail="SK_ID_CURR not found")
    
    # Retrieve data corresponding to SK_ID_CURR and make predictions
    data = fetch_data(sk_id)
    prediction = model.predict(data)
    
    return {"prediction": prediction.tolist()}

# Function to check if SK_ID_CURR exists in the database
def check_sk_id_exists(sk_id):
    return sk_id in df["SK_ID_CURR"].values

# Function to fetch data corresponding to SK_ID_CURR 
def fetch_data(sk_id):
    return df[df["SK_ID_CURR"] == sk_id]
