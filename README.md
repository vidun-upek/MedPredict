# MedPredict - Medical Cost Prediction

MedPredict is a web application that predicts the medical insurance charges for individuals based on factors such as age, BMI, smoking status, and more. This project uses machine learning to provide accurate predictions to users and is built using **Streamlit** for the user interface and **RandomForestRegressor** for the model.


## Model Performance

The final model performance is as follows:
- **R-squared**: 0.86
- **Root Mean Squared Error (RMSE)**: $300

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MedPredict.git

2. **Navigate into the project folder**:
    cd MedPredict

3. **Create a virtual environment and activate it**:
    python -m venv venv

On Windows:
venv\Scripts\activate


On macOS/Linux:
source venv/bin/activate

4.**Install the required libraries**:
    pip install -r requirements.txt

5.**Run the Streamlit app**:
    streamlit run app.py
