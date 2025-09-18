# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join('models', 'model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Features expected by the model
FEATURES = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
    'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
    'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# Mappings for categorical features
ENCODINGS = {
    "Sex": {"Male": 1, "Female": 0},
    "Exercise angina": {"Yes": 1, "No": 0},
    "Chest pain type": {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal": 3,
        "Asymptomatic": 4
    },
    "Thallium": {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def user_input():
    if request.method == 'POST':
        # Case 1: CSV file uploaded
        if 'csvfile' in request.files and request.files['csvfile'].filename != '':
            file = request.files['csvfile']
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return render_template('input.html', error=f'Error reading CSV: {e}', features=FEATURES)

            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                return render_template('input.html', error=f'Missing columns in CSV: {missing}', features=FEATURES)

            X = df[FEATURES].values
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)

            results = [
                {'index': i, 'prediction': int(p), 'label': 'Presence' if p == 1 else 'Absence'}
                for i, p in enumerate(preds)
            ]
            return render_template('output.html', results=results)

        # Case 2: Single record form submission
        try:
            vals = []
            for f in FEATURES:
                v = request.form.get(f)
                if v is None or v.strip() == '':
                    return render_template('input.html', error=f'Missing value for {f}', features=FEATURES)

                v = v.strip()

                # Check if categorical mapping exists
                if f in ENCODINGS and v in ENCODINGS[f]:
                    vals.append(ENCODINGS[f][v])
                else:
                    vals.append(float(v))  # numeric conversion

            X = np.array(vals).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = int(model.predict(X_scaled)[0])
            label = 'Presence' if pred == 1 else 'Absence'
            results = [{'index': 0, 'prediction': pred, 'label': label}]
            return render_template('output.html', results=results)
        except Exception as e:
            return render_template('input.html', error=f'Error processing input: {e}', features=FEATURES)

    # GET request
    return render_template('input.html', features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True)
