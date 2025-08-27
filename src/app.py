from flask import Flask, request, render_template_string
import pandas as pd
import pickle

# Load trained model once
model = pickle.load(open("heart_pipeline.pkl","rb"))

app = Flask(__name__)

# HTML form template
html_form = """
<!doctype html>
<title>Heart Disease Prediction</title>
<h2>Enter Patient Data</h2>
<form method="POST">
  Age: <input type="number" name="age" value="63"><br>
  Sex (1=Male, 0=Female): <input type="number" name="sex" value="1"><br>
  Chest Pain Type (0-3): <input type="number" name="cp" value="3"><br>
  Resting BP: <input type="number" name="trestbps" value="145"><br>
  Cholesterol: <input type="number" name="chol" value="233"><br>
  Fasting Blood Sugar (1/0): <input type="number" name="fbs" value="1"><br>
  Rest ECG (0-2): <input type="number" name="restecg" value="0"><br>
  Max Heart Rate: <input type="number" name="thalach" value="150"><br>
  Exercise Induced Angina (1/0): <input type="number" name="exang" value="0"><br>
  Oldpeak: <input type="text" name="oldpeak" value="2.3"><br>
  Slope (0-2): <input type="number" name="slope" value="0"><br>
  CA (0-3): <input type="number" name="ca" value="0"><br>
  Thal (1-3): <input type="number" name="thal" value="1"><br><br>
  <input type="submit" value="Predict">
</form>
{% if prediction is not none %}
<h3>Prediction: {{ prediction }}</h3>
<h4>Probability of Heart Disease: {{ proba }}</h4>
{% endif %}
"""

# Columns order
columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal']

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

@app.route("/", methods=["GET","POST"])
def home():
    prediction = None
    proba = None
    if request.method == "POST":
        # Extract features
        features = [safe_float(request.form.get(c), 0) for c in columns]
        features_df = pd.DataFrame([features], columns=columns)

        # Predict
        prediction = int(model.predict(features_df)[0])
        proba = float(model.predict_proba(features_df)[:,1][0])
    return render_template_string(html_form, prediction=prediction, proba=proba)

if __name__ == "__main__":
    app.run(debug=True)
