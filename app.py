import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('medpredict_model.pkl')

# Set the title of the app
st.title('MedPredict - Medical Cost Predictor')

age = st.number_input('Age', min_value=18, max_value=100)

sex = st.radio('Sex', ['male', 'female'])

bmi = st.number_input('BMI (Body Mass Index)')

children = st.slider('Number of Children', 0, 5)

smoker = st.radio('Are you a smoker?', ['yes', 'no'])

region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Create a button to trigger the prediction
if st.button('Predict My Cost'):
    # Inside the button logic, process the input and make the prediction

    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Make the prediction (log-transformed)
    prediction_log = model.predict(input_data)

    # Invert the log transformation to get the actual medical cost
    prediction_cost = np.exp(prediction_log[0])  # Invert the log for the first prediction

    # Display the predicted result
    st.success(f'Predicted Medical Cost: ${prediction_cost:,.2f}')
