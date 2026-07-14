import pickle
from pathlib import Path

from flask import Flask, render_template, request

from Training.train_model import preprocess_input, train_and_evaluate_models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "saved_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

app = Flask(__name__)
app.config["SECRET_KEY"] = "smart-lender-project"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    train_and_evaluate_models()

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/")
def home():
    """Render the landing page with an overview of the project."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Collect applicant details, predict the loan decision and display the result."""
    if request.method == "POST":
        input_data = {
            "Gender": request.form.get("Gender", "Male"),
            "Married": request.form.get("Married", "No"),
            "Dependents": request.form.get("Dependents", "0"),
            "Education": request.form.get("Education", "Graduate"),
            "Self_Employed": request.form.get("Self_Employed", "No"),
            "ApplicantIncome": float(request.form.get("ApplicantIncome", 0)),
            "CoapplicantIncome": float(request.form.get("CoapplicantIncome", 0)),
            "LoanAmount": float(request.form.get("LoanAmount", 0)),
            "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term", 360)),
            "Credit_History": float(request.form.get("Credit_History", 1)),
            "Property_Area": request.form.get("Property_Area", "Urban"),
        }
        scaled_input = preprocess_input(input_data, scaler=scaler)
        prediction = int(model.predict(scaled_input)[0])
        result_text = "Loan Approved" if prediction == 1 else "Loan Rejected"
        return render_template("submit.html", result=result_text)

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
