import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Global variables to hold our models (so we can access them in routes)
score_model = None
passfail_model = None
encoders = {}
data_loaded = False

# -----------------------------------------
# LOAD DATA & TRAIN MODEL (Safely)
# -----------------------------------------
try:
    # 1. Setup Path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "sample.csv")

    print(f"Attempting to load data from: {csv_path}")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # 2. Initialize Encoders
        le_gender = LabelEncoder()
        le_parent_edu = LabelEncoder()
        le_internet = LabelEncoder()
        le_extra = LabelEncoder()
        le_passfail = LabelEncoder()

        # 3. Fit Encoders
        df['Gender'] = le_gender.fit_transform(df['Gender'])
        df['Parental_Education_Level'] = le_parent_edu.fit_transform(df['Parental_Education_Level'])
        df['Internet_Access_at_Home'] = le_internet.fit_transform(df['Internet_Access_at_Home'])
        df['Extracurricular_Activities'] = le_extra.fit_transform(df['Extracurricular_Activities'])
        df['Pass_Fail'] = le_passfail.fit_transform(df['Pass_Fail'])

        # Save encoders to dictionary for easy access later
        encoders = {
            'Gender': le_gender,
            'Parental_Education_Level': le_parent_edu,
            'Internet_Access_at_Home': le_internet,
            'Extracurricular_Activities': le_extra,
            'Pass_Fail': le_passfail
        }

        # 4. Features & Targets
        X = df[['Gender','Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores',
                'Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']]
        y_score = df['Final_Exam_Score']
        y_passfail = df['Pass_Fail']

        # 5. Train Models
        score_model = RandomForestRegressor(n_estimators=50, random_state=42)
        score_model.fit(X, y_score)

        passfail_model = RandomForestClassifier(n_estimators=50, random_state=42)
        passfail_model.fit(X, y_passfail)

        data_loaded = True
        print("✅ Models trained successfully!")
    else:
        print(f"❌ ERROR: File not found at {csv_path}")

except Exception as e:
    print(f"❌ CRITICAL ERROR during training: {e}")

# -----------------------------------------
# ROUTES
# -----------------------------------------
@app.route("/")
def home():
    if data_loaded:
        return "✅ Flask app is running! Models are trained and ready."
    else:
        return "⚠️ App is running, but DATA FILE WAS NOT FOUND. Check that 'data/sample.csv' exists in GitHub."

@app.route("/predict", methods=["POST"])
def predict():
    if not data_loaded:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data = request.get_json()

        # Helper function to safely encode inputs
        def safe_encode(key, value):
            if value not in encoders[key].classes_:
                raise ValueError(f"Unknown value '{value}' for {key}")
            return encoders[key].transform([value])[0]

        # Prepare Input Features
        features = [[
            safe_encode('Gender', data['Gender']),
            float(data['Study_Hours_per_Week']),
            float(data['Attendance_Rate']),
            float(data['Past_Exam_Scores']),
            safe_encode('Parental_Education_Level', data['Parental_Education_Level']),
            safe_encode('Internet_Access_at_Home', data['Internet_Access_at_Home']),
            safe_encode('Extracurricular_Activities', data['Extracurricular_Activities'])
        ]]

        # Predict
        final_score = score_model.predict(features)[0]
        passfail_idx = passfail_model.predict(features)[0]
        passfail_label = encoders['Pass_Fail'].inverse_transform([passfail_idx])[0]

        return jsonify({
            "Final_Exam_Score": round(final_score, 2),
            "Pass_Fail": passfail_label
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

