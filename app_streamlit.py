import streamlit as st
import mlflow.pyfunc
import pandas as pd

# Load the MLFlow model
logged_model = 'runs:/b1f4405487db4f1db185eeed7734b8fe/model_logistic_regression'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Load the application_train.csv file
df = pd.read_csv('application_train.csv')

# User interface with Streamlit
st.title('Scoring Model')

# Input area for SK_ID_CURR of the client
st.header('Input')
sk_id_curr = st.number_input("Enter SK_ID_CURR:", min_value=0)  # Allowing only positive values

# Search for data corresponding to SK_ID_CURR in the dataframe
X_client = df[df['SK_ID_CURR'] == sk_id_curr]

# Prediction button
if st.button("Predict"):
    if X_client.empty:
        st.write(f"No data found for SK_ID_CURR: {sk_id_curr}")
    else:
        # Make predictions on client data
        y_pred_proba_client = loaded_model.predict(X_client)
        
        # Use the previously calculated optimal threshold for classification
        threshold = 0.5152  # Use the previously calculated optimal threshold
        y_pred_client = (y_pred_proba_client > threshold).astype(int)
        
        # Display the prediction for the client with SK_ID_CURR
        st.write(f"Prediction for client SK_ID_CURR = {sk_id_curr}: {y_pred_client}")