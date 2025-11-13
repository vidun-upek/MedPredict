import joblib
import pandas as pd
import numpy as np

# Load the trained model from the .pkl file
model = joblib.load('medpredict_model.pkl')

# Print the model to check if it has been loaded correctly
print("Model Loaded:", model)

# Sample input data (match the format the model expects)
input_data = pd.DataFrame({
    'age': [30],              # Example age
    'sex': ['male'],          # Example sex
    'bmi': [28],              # Example BMI
    'children': [2],          # Example number of children
    'smoker': ['yes'],        # Example smoker status
    'region': ['southeast']   # Example region
})

# Make a prediction (log-transformed)
prediction_log = model.predict(input_data)

# Invert the log transformation to get actual charges
prediction_cost = np.exp(prediction_log[0])  # Invert the log for the first prediction

print(f'Predicted Medical Cost: ${prediction_cost:,.2f}')