# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = 'insurance_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    age = [x for x in request.form.values()]
    bmi = [y for y in request.form.values()]
    smoker = [z for z in request.form.values()]
    
    final_features = pd.DataFrame([age,bmi,smoker],columns=["age","bmi","smoker"])
    final_features["smoker"].replace(to_replace={'no':0,'yes':1,'No':0,'Yes':1,'NO':0,'YES':1},inplace=True)
    
    # Make prediction
    prediction = np.exp2(model.predict(final_features))
    output = round(prediction[0],2)

    return render_template('index.html', prediction_text='Price of Insurance is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)