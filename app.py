from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("model/house_price_model.joblib")
model_columns = joblib.load("model/model_columns.joblib")  # keep your current spelling

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        input_data = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "YearBuilt": int(request.form["YearBuilt"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode Neighborhood
        input_df = pd.get_dummies(input_df)

        # Align with training columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ₦{prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)