import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os

# --- Load dataset ---
df = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/JANAKI AI PROJECT/Heart_Disease_Prediction.csv')

# Encode target column: Absence -> 0, Presence -> 1
df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# Features / target
X = df.drop(['Heart Disease'], axis=1)
y = df['Heart Disease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(random_state=2)
model.fit(X_train_scaled, y_train)

# Save model + scaler
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('✅ Saved model to models/model.pkl and scaler to models/scaler.pkl')

# Optional: check accuracy
from sklearn.metrics import accuracy_score, classification_report
preds = model.predict(X_test_scaled)
print('Accuracy on test set:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
