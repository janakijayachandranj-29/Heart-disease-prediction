# 🫀 Heart Disease Prediction - ML Deployment

This project is a **Machine Learning Deployment** built using **Flask (Python)** for the backend and **HTML/CSS** for the frontend.  
The trained model predicts whether a patient has **Heart Disease** (0 = Absence, 1 = Presence) based on medical features.

---

## 📂 Project Structure

Heart-Disease-Prediction/
│
├── models/
│ ├── model.pkl # Trained Decision Tree model
│ ├── scaler.pkl # StandardScaler for preprocessing
│
├── templates/
│ ├── welcome.html # Welcome page
│ ├── input.html # User input page (manual or CSV upload)
│ ├── output.html # Prediction results page
│
├── train_save.py # Script to train and save the model
├── app.py # Flask web application
├── README.md # Documentation
└── Heart_Disease_Prediction.csv # Dataset (not included in repo)

yaml
Copy code

---

## ⚙️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
Set up virtual environment

bash
Copy code
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac
Install dependencies

bash
Copy code
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy code
flask
numpy
pandas
scikit-learn
Train the model and save artifacts

bash
Copy code
python train_save.py
This generates:

models/model.pkl

models/scaler.pkl

Run Flask app

bash
Copy code
python app.py
Open in browser:

cpp
Copy code
http://127.0.0.1:5000/
🖼️ Web Pages
Welcome Page → welcome.html

Input Page → input.html

Fill out patient details OR upload a CSV file.

Output Page → output.html

Displays prediction:

0 = Absence

1 = Presence

📊 Dataset Details
Dataset: Heart_Disease_Prediction.csv

Target column: Heart Disease (Absence → 0, Presence → 1)

Features include:

Age

Sex

Chest pain type

BP

Cholesterol

FBS over 120

EKG results

Max HR

Exercise angina

ST depression

Slope of ST

Number of vessels fluro

Thallium

🚀 Future Enhancements
Deploy on Heroku / AWS / Azure

Add Bootstrap/React frontend

Integrate advanced ML models (Random Forest, XGBoost)

Provide API endpoints for external usage

👩‍💻 Author
Janaki Jayachandran
Machine Learning Deployment Project

yaml
Copy code

---

👉 Do you also want me to create a **`requirements.txt`** for you (so you can `pip install -r requirements.txt` directly)?






Ask ChatGPT
