from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------
# Initialize Flask app
# ---------------------------------------
app = Flask(__name__)

# ---------------------------------------
# Load trained model and columns
# ---------------------------------------
model = joblib.load("house_price_rf_model.pkl")
columns = joblib.load("columns.pkl")

# ---------------------------------------
# Home route
# ---------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------------
# Prediction route
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------------------------------------
        # Read form inputs
        # ---------------------------------------
        posted_by = request.form["POSTED_BY"]
        bhk_or_rk = request.form["BHK_OR_RK"]

        bhk_no = int(request.form["BHK_NO"])
        square_ft = float(request.form["SQUARE_FT"])

        latitude = float(request.form["LATITUDE"])
        longitude = float(request.form["LONGITUDE"])

        under_construction = int(request.form["UNDER_CONSTRUCTION"])
        ready_to_move = int(request.form["READY_TO_MOVE"])
        resale = int(request.form["RESALE"])
        rera = int(request.form["RERA"])

        # ---------------------------------------
        # Feature engineering (same as training)
        # ---------------------------------------
        bhk_density = bhk_no / square_ft if square_ft != 0 else 0

        # NOTE:
        # LOCATION_CLUSTER was created using KMeans during training.
        # Since clustering model is not loaded in Flask,
        # we use a safe default cluster = 0
        location_cluster = 0

        # ---------------------------------------
        # Create input DataFrame
        # ---------------------------------------
        input_data = {
            "POSTED_BY": posted_by,
            "UNDER_CONSTRUCTION": under_construction,
            "RERA": rera,
            "BHK_NO.": bhk_no,
            "BHK_OR_RK": bhk_or_rk,
            "SQUARE_FT": square_ft,
            "READY_TO_MOVE": ready_to_move,
            "RESALE": resale,
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "BHK_DENSITY": bhk_density,
            "LOCATION_CLUSTER": location_cluster
        }

        input_df = pd.DataFrame([input_data])

        # ---------------------------------------
        # One-hot encode
        # ---------------------------------------
        input_df = pd.get_dummies(input_df)

        # ---------------------------------------
        # Align with training columns
        # ---------------------------------------
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # ---------------------------------------
        # Predict
        # ---------------------------------------
        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated House Price: ₹ {round(prediction, 2)} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error occurred: {str(e)}"
        )

# ---------------------------------------
# Run app
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
