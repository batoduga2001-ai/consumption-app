from flask import Flask, request, render_template_string
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("telco_churn_model.pkl")

app = Flask(__name__)

# =========================
# SIMPLE HTML UI
# =========================
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Telco Churn Predictor</title>
</head>
<body>
    <h2>Telco Customer Churn Prediction</h2>

    <form method="POST">
        <label>Tenure:</label>
        <input type="number" name="tenure" required><br><br>

        <label>Monthly Charges:</label>
        <input type="number" step="0.01" name="MonthlyCharges" required><br><br>

        <label>Total Charges:</label>
        <input type="number" step="0.01" name="TotalCharges" required><br><br>

        <button type="submit">Predict</button>
    </form>

    <h3>{{ prediction }}</h3>
</body>
</html>
"""

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        tenure = float(request.form["tenure"])
        monthly = float(request.form["MonthlyCharges"])
        total = float(request.form["TotalCharges"])

        # IMPORTANT: match training feature order
        input_data = pd.DataFrame([[tenure, monthly, total]],
                                  columns=["tenure", "MonthlyCharges", "TotalCharges"])

        result = model.predict(input_data)[0]

        prediction = "⚠ Customer WILL CHURN" if result == 1 else "✅ Customer WILL STAY"

    return render_template_string(html, prediction=prediction)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)