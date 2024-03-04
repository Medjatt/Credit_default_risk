import streamlit as st
import requests
import matplotlib.pyplot as plt

# User interface with Streamlit
st.title('Scoring Model')

# Input area for SK_ID_CURR of the client
st.header('Input')
sk_id_curr = st.number_input("Enter client SK_ID_CURR:", min_value=0)  # Allowing only positive values

# Prediction button
if st.button("Predict"):
    # Endpoint of the deployed Azure API
    endpoint = "https://modelscore.azurewebsites.net/predict/"

    # Make a GET request to the Azure API with the SK_ID_CURR
    response = requests.get(endpoint + str(sk_id_curr))

    if response.status_code == 200:
        # Extract prediction from the response
        prediction_data = response.json()
        
        # Display the prediction for the client with SK_ID_CURR
        st.write(f"Prediction for client SK_ID_CURR = {sk_id_curr}:")
        st.write(f"Probability of failure: {prediction_data['probability_of_failure']:.4f}")
        st.write(f"Class: {prediction_data['class']}")

        # Plot pie chart representing probability of failure
        prob_failure = prediction_data['probability_of_failure']
        prob_success = 1 - prob_failure

        # Data to plot
        labels = ['Probability of failure', 'Probability of non-failure']
        sizes = [prob_failure, prob_success]
        colors = ['red', 'green']

        # Plot
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%')

        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')  

        # Show the plot
        st.pyplot(fig)
    else:
        st.write(f"Client ID = {sk_id_curr} not found in the database.")