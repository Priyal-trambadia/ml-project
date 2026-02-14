import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(BASE_DIR, "data", "sample.csv"),
    os.path.join(BASE_DIR, "sample.csv"),
    "data/sample.csv",
    "sample.csv"
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path:
    try:
        df = pd.read_csv(csv_path)
        
        # Train Encoders
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

        encoders = {
            'Gender': le_gender,
            'Parental_Education_Level': le_parent_edu,
            'Internet_Access_at_Home': le_internet,
            'Extracurricular_Activities': le_extra,
            'Pass_Fail': le_passfail
        }

        # Train Models
        X = df[['Gender','Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores',
                'Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']]
        y_score = df['Final_Exam_Score']
        y_passfail = df['Pass_Fail']

        score_model = RandomForestRegressor(n_estimators=50, random_state=42)
        score_model.fit(X, y_score)

        passfail_model = RandomForestClassifier(n_estimators=50, random_state=42)
        passfail_model.fit(X, y_passfail)

        data_loaded = True
        print("✅ Models trained successfully!")
    except Exception as e:
        print(f"❌ ERROR: {e}")

# -----------------------------------------
# THE WEBSITE DESIGN (HTML)
# -----------------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Exam Predictor</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background: #f4f4f9; }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { width: 100%; padding: 10px; background: #28a745; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #218838; }
        #result { margin-top: 20px; padding: 15px; background: #fff; border-left: 5px solid #28a745; display: none; }
    </style>
</head>
<body>
    <h1>🎓 Student Performance Predictor</h1>
    
    <div class="form-group">
        <label>Gender</label>
        <select id="Gender"><option>Male</option><option>Female</option></select>
    </div>
    <div class="form-group">
        <label>Study Hours per Week</label>
        <input type="number" id="Study_Hours" value="10">
    </div>
    <div class="form-group">
        <label>Attendance Rate (%)</label>
        <input type="number" id="Attendance" value="85">
    </div>
    <div class="form-group">
        <label>Past Exam Scores</label>
        <input type="number" id="Past_Scores" value="70">
    </div>
    <div class="form-group">
        <label>Parental Education Level</label>
        <select id="Parent_Edu"><option>High School</option><option>College</option><option>Postgraduate</option></select>
    </div>
    <div class="form-group">
        <label>Internet Access at Home</label>
        <select id="Internet"><option>Yes</option><option>No</option></select>
    </div>
    <div class="form-group">
        <label>Extracurricular Activities</label>
        <select id="Extra"><option>Yes</option><option>No</option></select>
    </div>

    <button onclick="predict()">Predict Score</button>

    <div id="result">
        <h3>Prediction Results:</h3>
        <p><strong>Predicted Score:</strong> <span id="score_out">-</span></p>
        <p><strong>Pass/Fail Status:</strong> <span id="pass_out">-</span></p>
    </div>

    <script>
        async function predict() {
            const data = {
                Gender: document.getElementById('Gender').value,
                Study_Hours_per_Week: document.getElementById('Study_Hours').value,
                Attendance_Rate: document.getElementById('Attendance').value,
                Past_Exam_Scores: document.getElementById('Past_Scores').value,
                Parental_Education_Level: document.getElementById('Parent_Edu').value,
                Internet_Access_at_Home: document.getElementById('Internet').value,
                Extracurricular_Activities: document.getElementById('Extra').value
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            document.getElementById('result').style.display = 'block';
            if(result.error) {
                document.getElementById('score_out').innerText = "Error: " + result.error;
            } else {
                document.getElementById('score_out').innerText = result.Final_Exam_Score;
                document.getElementById('pass_out').innerText = result.Pass_Fail;
            }
        }
    </script>
</body>
</html>
"""

# -----------------------------------------
# ROUTES
# -----------------------------------------
@app.route("/")
def home():
    if data_loaded:
        return render_template_string(html_template)
    else:
        return "⚠️ App is running, but DATA FILE WAS NOT FOUND. Check your logs!"

@app.route("/predict", methods=["POST"])
def predict():
    if not data_loaded:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        
        # Helper to encode inputs safely
        def safe_encode(key, value):
            try:
                return encoders[key].transform([value])[0]
            except:
                # Fallback if value (like 'Male') isn't found exactly, try lowercase/trim
                # This is a basic fallback, strict matching is better for production
                return encoders[key].transform([encoders[key].classes_[0]])[0] 

        features = [[
            safe_encode('Gender', data['Gender']),
            float(data['Study_Hours_per_Week']),
            float(data['Attendance_Rate']),
            float(data['Past_Exam_Scores']),
            safe_encode('Parental_Education_Level', data['Parental_Education_Level']),
            safe_encode('Internet_Access_at_Home', data['Internet_Access_at_Home']),
            safe_encode('Extracurricular_Activities', data['Extracurricular_Activities'])
        ]]

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
    app.run(debug=True, host="0.0.0.0", port=5000)
