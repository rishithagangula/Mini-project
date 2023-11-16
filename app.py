import numpy as np
import pandas as pd
from  joblib import *
import requests
from flask import Flask, render_template,request
from flask_ngrok import run_with_ngrok

app = Flask(__name__)


def load_models():
    model1 = load("RFS.joblib")   #random forest model
    model2=load("RFS1.joblib")    #random forest model
    return model1, model2

model1, model2 = load_models()

# Define air quality intervals and labels
intervals = [0, 50, 100, 200, 300, 400, 2500]
labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# Define symptoms, diseases, and precautions
symptoms = ['Air quality is good in this range, and most people will not experience any symptoms',
            'Experience mild symptoms like coughing or throat irritation',
            'Coughing, Shortness of breath, or Chest discomfort',
            'Respiratory conditions, including shortness of breath, coughing, and chest tightness',
            'Respiratory symptoms for most individuals, including coughing, throat irritation, and difficulty breathing',
            'Severe respiratory distress for everyone, even healthy individuals']

diseases = ['None',
            'Mild symptoms of allergies and sinusitis. Asthma symptoms can be aggravated',
            'Respiratory Infections',
            'Exacerbation of asthma, Chronic Obstructive Pulmonary, and increased cardiovascular diseases',
            'Exacerbation of respiratory diseases, cardiovascular issues, and general health risks',
            'Severe risk of heart attacks and respiratory diseases']

precautions = ['Enjoy outdoor activities and open-air exercise',
              'Sensitive individuals should reduce outdoor activities during periods of elevated AQI',
              'People with asthma and heart conditions should limit outdoor activities. Consider using air purifiers indoors',
              'Minimize outdoor activities, especially for children, the elderly, and individuals with pre-existing conditions. Use N95 or equivalent masks if outdoor exposure is unavoidable. Create a clean indoor environment with air purifiers and keep windows closed',
              'Stay indoors as much as possible, and keep windows and doors sealed. Use air purifiers with HEPA filters. Vulnerable populations, like children and the elderly, should take extra precautions',
              'Wear N95 or higher-rated masks if you must go outside,although its best to avoid outdoor exposure. Seek immediate medical attention for severe symptoms.']

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    model1_info = None
    model2_info = None

    if request.method == 'POST':
        try:
            input_data = float(request.form['aqiInput'])

            # Check if the input value is within the expected range
            if 0 <= input_data <= 2500:
                # Make predictions using both models
                prediction1 = [input_data]  # Wrap the scalar value in a list
               

                def get_info(prediction, model_name):
                    air_quality = pd.cut(prediction, bins=intervals, labels=labels)
                    symptoms_info = pd.cut(prediction, bins=intervals, labels=symptoms)
                    disease_info = pd.cut(prediction, bins=intervals, labels=diseases)
                    precaution_info = pd.cut(prediction, bins=intervals, labels=precautions)
                    return {
                        'Model Name': model_name,
                        'Prediction': prediction[0],
                        'Air Quality Class': air_quality[0],
                        'Symptoms': symptoms_info[0],
                        'Diseases': disease_info[0],
                        'Precautions': precaution_info[0]
                    }

                model1_info = get_info(prediction1, 'Model 1')
                
                print("\nModel 1 Prediction:")
                print(model1_info)
               
            else:
                print("AQI value is out of range (0-2500).")
        except Exception as e:
            return "An error occurred: " + str(e)

    # Handle the GET request to display the form
    return render_template('predict.html', model1_info=model1_info)


if __name__ == "__main__":
    app.debug = True  # Enable debug mode
    app.run()
