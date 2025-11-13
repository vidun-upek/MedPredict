import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('insurance.csv')

# Define features (X) and target (y)
X = df.drop('charges', axis=1)  # Drop the 'charges' column to use other columns as features
y = np.log(df['charges'])  # Apply log transformation to 'charges'

# Identify feature types
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Apply StandardScaler to numeric features
        ('cat', OneHotEncoder(), categorical_features)  # Apply OneHotEncoder to categorical features
    ])


# Create full model pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing step
    ('model', RandomForestRegressor())  # Machine learning model (Random Forest)
])


# Split the data into train and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
full_pipeline.fit(X_train, y_train)


# Make predictions on the test set
y_pred_log = full_pipeline.predict(X_test)

# Invert the log transformation to get actual charges
y_pred_actual = np.exp(y_pred_log)
y_test_actual = np.exp(y_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared: {r2}')


# Save the trained model to a file
joblib.dump(full_pipeline, 'medpredict_model.pkl')
