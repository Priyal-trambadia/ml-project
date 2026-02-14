import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------
# INITIALIZE APP
# -----------------------------------------
app = Flask(__name__)
CORS(app)

# Global variables
score_model = None
passfail_model = None
encoders = {}
data_loaded = False

# -----------------------------------------
# SMART DATA LOADER
# -----------------------------------------
print("🔍 STARTING APP: Looking for data file...")

# Get the current folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List of all possible places the file might be
possible_paths = [
    os.path.join(BASE_DIR, "data", "sample.csv"),    # 1. Standard: inside 'data' folder
    os.path.join(BASE_DIR, "sample.csv"),            # 2. Backup: right next to app.py
    "data/sample.csv",                               # 3. Relative path
    "sample.csv"                                     # 4. Simple filename
]

csv_path = None

# Loop through paths to find the file
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ FOUND FILE AT: {path}")
        csv_path = path
        break

# -----------------------------------------
# LOAD DATA & TRAIN MODELS
# -----------------------------------------
if csv_path:
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ SUCCESS: Loaded {len(df)} rows.")

        # --- 1. Label Encoding ---
        le_gender = LabelEncoder()
        le_parent_edu = LabelEncoder()
        le_internet = LabelEncoder()
        le_extra = LabelEncoder()
        le_passfail = LabelEncoder()

        df['Gender'] = le_gender.fit_transform(df['Gender'])
        df['Parental_Education_Level'] = le_parent_edu.fit_transform(df['Parental_Education_Level'])
        df['Internet_Access_at_Home'] = le_internet.fit_transform(df['Internet_Access_at_Home'])
        df['Extracurricular_Activities'] = le_extra.fit_transform(df['Extracurricular_Activities'])
        df['Pass_Fail'] = le_passfail.fit_transform(df['Pass_Fail'])

        # Save encoders so we can use them in the /predict route
        encoders = {
            'Gender': le_gender,
            'Parental_Education_Level': le_parent_edu,
            'Internet_Access_at_Home': le_internet,
            'Extracurricular_Activities': le_extra,
            'Pass_Fail': le_passfail
        }

        # --- 2. Features & Targets ---
        X = df[['Gender','Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores',
                'Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']]
        
        y_score = df['Final_Exam_Score']
        y_passfail = df['Pass_Fail']

        # --- 3. Train Models ---
        score_model = RandomForestRegressor(n_estimators=50, random_state=42)
        score_model.fit(X, y_score)

        passfail_model = RandomForestClassifier(n_estimators=50, random_state=42)
        passfail_model.fit(X, y_passfail)

        data_loaded = True
        print("✅ Models trained successfully!")

    except Exception as e:
        print(f"❌ ERROR processing data: {e}")
else:
    print("❌ FATAL ERROR: File 'sample.csv' NOT found in any expected location.")
    # Print directory contents to logs for debugging
    print("📂 Current Directory Contents:", os.listdir(BASE_DIR))
    if os.path.exists(os.path.join(BASE_DIR, "data")):
        print("📂 Contents of 'data' folder:", os.listdir(os.path.join(BASE_DIR, "data")))

# -----------------------------------------
# ROUTES
# -----------------------------------------

@app.route("/")
def home():
    if data_loaded:
        return "✅ Flask App is Running! Models are Ready."
    else:
        # If this appears, check your Render Logs for the '📂 Current Directory Contents' line
        return "⚠️ App is running, but DATA FAILED TO LOAD. Check Render Logs for details."

@app.route("/predict", methods=["POST"])
def predict():
    if not data_loaded:
        return jsonify({"error": "Models are not loaded because data file was missing."}), 500

    try:
        data = request.get_json()

        # Helper to encode inputs safely
        def safe_encode(key, value):
            if value not in encoders[key].classes_:
                raise ValueError(f"Invalid value '{value}' for {key}")
            return encoders[key].transform([value])[0]

        # Prepare features
        features = [[
            safe_encode('Gender', data['Gender']),
            float(data['Study_Hours_per_Week']),
            float(data['Attendance_Rate']),
            float(data['Past_Exam_Scores']),
            safe_encode('Parental_Education_Level', data['Parental_Education_Level']),
            safe_encode('Internet_Access_at_Home', data['Internet_Access_at_Home']),
            safe_encode('Extracurricular_Activities', data['Extracurricular_Activities'])
        ]]

        # Make predictions
        final_score = score_model.predict(features)[0]
        passfail_idx = passfail_model.predict(features)[0]
        passfail_label = encoders['Pass_Fail'].inverse_transform([passfail_idx])[0]

        return jsonify({
            "Final_Exam_Score": round(final_score, 2),
            "Pass_Fail": passfail_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------
# RUN LOCALLY
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
