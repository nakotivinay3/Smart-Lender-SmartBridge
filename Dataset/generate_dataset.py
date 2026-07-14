import os
import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(output_path=None) -> pd.DataFrame:
    """Create a synthetic loan eligibility dataset with realistic fields and missing values."""
    rng = np.random.default_rng(42)
    n_rows = 400

    data = {
        "Gender": rng.choice(["Male", "Female"], size=n_rows, p=[0.65, 0.35]),
        "Married": rng.choice(["Yes", "No"], size=n_rows, p=[0.58, 0.42]),
        "Dependents": rng.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.45, 0.25, 0.20, 0.10]),
        "Education": rng.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.75, 0.25]),
        "Self_Employed": rng.choice(["No", "Yes"], size=n_rows, p=[0.88, 0.12]),
        "ApplicantIncome": np.clip(rng.normal(5400, 2200, n_rows), 1000, 30000).astype(int),
        "CoapplicantIncome": np.clip(rng.normal(1800, 1700, n_rows), 0, 15000).astype(int),
        "LoanAmount": np.clip(rng.normal(140, 55, n_rows), 50, 600).astype(int),
        "Loan_Amount_Term": rng.choice([180, 360, 480, 300], size=n_rows, p=[0.08, 0.78, 0.07, 0.07]),
        "Credit_History": rng.choice([1.0, 0.0], size=n_rows, p=[0.75, 0.25]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], size=n_rows, p=[0.35, 0.40, 0.25]),
    }

    df = pd.DataFrame(data)
    df["Loan_Status"] = np.where(
        (df["Credit_History"] == 1.0)
        & ((df["ApplicantIncome"] + df["CoapplicantIncome"]) < 12000)
        & (df["LoanAmount"] < 300),
        "Y",
        "N",
    )

    # Introduce realistic missing values.
    for column in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "LoanAmount"]:
        mask = rng.random(n_rows) < 0.08
        df.loc[mask, column] = np.nan

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "loan_prediction.csv"
    else:
        output_path = Path(output_path)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generate_dataset()
    print("Dataset generated successfully at Dataset/loan_prediction.csv")
