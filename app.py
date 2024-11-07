from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load only the trained model
model = joblib.load('diabetes_rf_model_with_features.pkl')

# Manually specify the feature names based on the original training dataset
feature_names = [
    'age', 'bmi', 'blood_glucose_level', 'HbA1c_level', 
]

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    data = {feature: request.form.get(feature, 0) for feature in feature_names}

    # Convert input data to DataFrame with correct structure
    input_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result on the HTML page
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return render_template('index.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
