from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/stress_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    inputs = [
        int(request.form["sleep_quality"]),
        int(request.form["headache_frequency"]),
        int(request.form["academic_performance"]),
        int(request.form["study_load"]),
        int(request.form["extracurricular_activity"])
    ]

    scaled = scaler.transform([inputs])
    prediction = model.predict(scaled)[0]

    stress_map = {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    }

    return render_template(
        "index.html",
        prediction=prediction,
        stress_label=stress_map[prediction]
    )

if __name__ == "__main__":
    app.run(debug=True)
