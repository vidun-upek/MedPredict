
MedPredict - Medical Cost Prediction 🏥
MedPredict is a web application designed to predict medical insurance charges for individuals. It utilizes machine learning to offer accurate cost estimations based on several personal and health-related factors.

💻 Key Technologies
User Interface (UI): Built using Streamlit.

Machine Learning Model: The predictions are powered by a RandomForestRegressor.

📈 Prediction Factors
The application takes into account various factors to generate a prediction, including:

Age

BMI (Body Mass Index)

Smoking status

And more

✨ Model Performance
The final machine learning model exhibits the following performance metrics:

R-squared: 0.86

Root Mean Squared Error (RMSE): $$300$

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
