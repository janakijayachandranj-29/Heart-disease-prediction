from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
with open(MODEL_PATH, 'rb') as f:
model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
scaler = pickle.load(f)


# Column names expected by the model (order matters for CSV upload)
FEATURES = [
'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
'Slope of ST', 'Number of vessels fluro', 'Thallium'
]


@app.route('/')
def welcome():
return render_template('welcome.html')


@app.route('/input', methods=['GET', 'POST'])
def user_input():
if request.method == 'POST':
# If CSV uploaded
if 'csvfile' in request.files and request.files['csvfile'].filename != '':
file = request.files['csvfile']
try:
df = pd.read_csv(file)
except Exception as e:
return render_template('input.html', error=f'Error reading CSV: {e}')


# If CSV contains all features, use them. Otherwise expect same column order.
missing = [c for c in FEATURES if c not in df.columns]
if missing:
return render_template('input.html', error=f'Missing columns in CSV: {missing}')


X = df[FEATURES].values
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)


# Prepare results for output page
results = [{'index': i, 'prediction': int(p), 'label': 'Presence' if p==1 else 'Absence'} for i,p in enumerate(preds)]
return render_template('output.html', results=results)


# Otherwise, single-record form submission
try:
vals = []
for f in FEATURES:
v = request.form.get(f)
if v is None or v == '':
return render_template('input.html', error=f'Missing value for {f}', features=FEATURES)
# convert to numeric (allow floats)
vals.append(float(v))
X = np.array(vals).reshape(1, -1)
X_scaled = scaler.transform(X)
pred = int(model.predict(X_scaled)[0])
label = 'Presence' if pred == 1 else 'Absence'
results = [{'index': 0, 'prediction': pred, 'label': label}]
return render_template('output.html', results=results)
except Exception as e:
return render_template('input.html', error=f'Error processing input: {e}', features=FEATURES)


# GET
return render_template('input.html', features=FEATURES)


if __name__ == '__main__':
app.run(debug=True)