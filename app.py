from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def extract_features(features):
    try:
        features = features.split(',')
        np_features = np.asarray(features, dtype=np.float32)
        return np_features.reshape(1, -1)
    except ValueError:
        return None

def predict_cancer(features):
    try:
        pred = model.predict(features)
        return pred[0]
    except Exception as e:
        print(f"Error predicting cancer: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', message=["Error: Model not loaded"])

    features = request.form['feature']
    np_features = extract_features(features)

    if np_features is None:
        return render_template('index.html', message=["Invalid input features"])

    pred = predict_cancer(np_features)

    if pred is None:
        return render_template('index.html', message=["Error predicting cancer"])

    message = ['Cancrouse' if pred == 1 else 'Not Cancrouse']
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)