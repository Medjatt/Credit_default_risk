from fastapi import FastAPI, Path, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Charger le modèle
model = joblib.load("model.joblib")

# Endpoint pour faire des prédictions avec SK_ID_CURR en tant que chemin dans l'URL
@app.get("/predict/{sk_id}")
def predict(sk_id: int = Path(..., title="SK_ID_CURR", description="L'identifiant unique du client")):
    # Valider l'entrée utilisateur (par exemple, vérifier si SK_ID_CURR existe dans la base de données)
    if not check_sk_id_exists(sk_id):
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé")
    
    # Récupérer les données correspondant à SK_ID_CURR et effectuer les prédictions
    data = fetch_data(sk_id)
    prediction = model.predict(data)
    
    return {"prediction": prediction.tolist()}

# Fonction pour vérifier si SK_ID_CURR existe dans la base de données
def check_sk_id_exists(sk_id):
    # Exemple de vérification dans un DataFrame pandas
    df = pd.read_csv("application_train.csv")  # Charger les données à partir d'un fichier CSV
    return sk_id in df["SK_ID_CURR"].values

# Fonction pour récupérer les données correspondant à SK_ID_CURR
def fetch_data(sk_id):
    # Exemple de récupération de données à partir d'un DataFrame pandas
    df = pd.read_csv("application_train.csv")  # Charger les données à partir d'un fichier CSV
    return df[df["SK_ID_CURR"] == sk_id]  # Retourner les lignes correspondant à SK_ID_CURR