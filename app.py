from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Flask app is running successfully!"

# -----------------------------------------
# LOAD DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "sample.csv")

df = pd.read_csv(csv_path)

# -----------------------------------------
# LABEL ENCODING
# -----------------------------------------
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

# -----------------------------------------
# FEATURES & TARGETS
# -----------------------------------------
X = df[['Gender','Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores',
        'Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']]

y_score = df['Final_Exam_Score']
y_passfail = df['Pass_Fail']

# -----------------------------------------
# TRAIN MODELS
# -----------------------------------------
score_model = RandomForestRegressor(n_estimators=50, random_state=42)
score_model.fit(X, y_score)

passfail_model = RandomForestClassifier(n_estimators=50, random_state=42)
passfail_model.fit(X, y_passfail)

# -----------------------------------------
# PREDICT ROUTE
# -----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Encode inputs
        gender = le_gender.transform([data['Gender']])[0]
        parent_edu = le_parent_edu.transform([data['Parental_Education_Level']])[0]
        internet = le_internet.transform([data['Internet_Access_at_Home']])[0]
        extra = le_extra.transform([data['Extracurricular_Activities']])[0]

        features = [[
            gender,
            float(data['Study_Hours_per_Week']),
            float(data['Attendance_Rate']),
            float(data['Past_Exam_Scores']),
            parent_edu,
            internet,
            extra
        ]]

        # Predict score and pass/fail
        final_score = score_model.predict(features)[0]
        passfail = passfail_model.predict(features)[0]
        passfail_label = le_passfail.inverse_transform([passfail])[0]

        return jsonify({
            "Final_Exam_Score": round(final_score, 2),
            "Pass_Fail": passfail_label
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------------------
# RUN APP
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
