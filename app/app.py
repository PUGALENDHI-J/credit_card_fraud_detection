from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if not file:
        return render_template("index.html", result="No file uploaded")

    # FIX: specify encoding
    data = pd.read_csv(
    file,
    encoding="latin1",
    sep=",",
    engine="python",
    on_bad_lines="skip"
)


    prediction = model.predict(data)

    fraud_count = sum(prediction)
    legit_count = len(prediction) - fraud_count

    return render_template(
        "index.html",
        result=f"Fraud Transactions: {fraud_count} | Legit Transactions: {legit_count}"
    )

if __name__ == "__main__":
    app.run(debug=True)
