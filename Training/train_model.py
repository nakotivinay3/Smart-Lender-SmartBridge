import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Dataset" / "loan_prediction.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
EDA_DIR = BASE_DIR / "Training" / "eda_outputs"
EDA_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLUMNS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
TARGET_COLUMN = "Loan_Status"

CATEGORY_MAPPINGS = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}
TARGET_MAPPING = {"Y": 1, "N": 0}


def load_dataset(path=None) -> pd.DataFrame:
    """Load the loan prediction dataset from the local CSV file."""
    file_path = path or DATASET_PATH
    return pd.read_csv(file_path)


def perform_eda(df: pd.DataFrame) -> None:
    """Generate and save a complete exploratory data analysis summary for the project."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["ApplicantIncome"], bins=20, kde=True)
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "income_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Loan_Status")
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "loan_status_count.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="Gender", ax=axes[0])
    axes[0].set_title("Gender Distribution")
    sns.countplot(data=df, x="Education", ax=axes[1])
    axes[1].set_title("Education Distribution")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "categorical_subplots.png")
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Gender", hue="Married")
    plt.title("Gender vs Married")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "gender_married.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Education", hue="Self_Employed")
    plt.title("Education vs Self Employed")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "education_self_employed.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Property_Area", y="Loan_Amount_Term")
    plt.title("Property Area vs Loan Amount Term")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "property_area_term.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.swarmplot(data=df, x="Property_Area", y="LoanAmount", hue="Loan_Status")
    plt.title("Property Area vs Loan Amount by Loan Status")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "swarmplot.png")
    plt.close()


def preprocess_features(df: pd.DataFrame):
    """Clean the dataset, fill missing values, encode categories and prepare features and target."""
    prepared_df = df.copy()

    for column in NUMERIC_COLUMNS:
        prepared_df[column] = prepared_df[column].fillna(prepared_df[column].mean())

    for column in CATEGORICAL_COLUMNS:
        prepared_df[column] = prepared_df[column].fillna(prepared_df[column].mode()[0])

    for column, mapping in CATEGORY_MAPPINGS.items():
        prepared_df[column] = prepared_df[column].astype(str).str.strip().str.title()
        prepared_df[column] = prepared_df[column].map(mapping)

    prepared_df[TARGET_COLUMN] = prepared_df[TARGET_COLUMN].map(TARGET_MAPPING)

    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    features = prepared_df[feature_columns]
    target = prepared_df[TARGET_COLUMN]
    return features, target


def build_feature_frame(input_data: dict) -> pd.DataFrame:
    """Convert incoming form values into a feature DataFrame with the same encoding used in training."""
    prepared = pd.DataFrame([input_data])

    # Convert numerical fields to float values.
    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(prepared[column].mean())

    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype(str).str.strip().str.title()
        prepared[column] = prepared[column].map(CATEGORY_MAPPINGS[column])

    return prepared[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit a StandardScaler on training data and transform both train and test splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    """Train the model, produce predictions and collect core evaluation metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, target_names=["Rejected", "Approved"]),
        "f1": f1_score(y_test, predictions),
    }


def train_and_evaluate_models():
    """Run the complete training workflow and save the best model and scaler."""
    df = load_dataset()
    perform_eda(df)
    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)

    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_estimators=120),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train_scaled, X_test_scaled, y_train_resampled, y_test)

    best_name = max(results, key=lambda model_name: results[model_name]["accuracy"])
    best_model = results[best_name]["model"]

    # Cross validation for the best model.
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_resampled, cv=5)

    with open(MODEL_DIR / "saved_model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

    with open(MODEL_DIR / "scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    summary = {
        "best_model": best_name,
        "best_accuracy": round(results[best_name]["accuracy"], 4),
        "cross_val_mean": round(float(cv_scores.mean()), 4),
        "cross_val_scores": [round(float(score), 4) for score in cv_scores],
        "results": {name: {"accuracy": round(metrics["accuracy"], 4), "f1": round(metrics["f1"], 4)} for name, metrics in results.items()},
    }

    print("Training completed successfully.")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {summary['best_accuracy']}")
    print(f"Cross-validation mean: {summary['cross_val_mean']}")
    return summary


def preprocess_input(input_data: dict, scaler=None) -> np.ndarray:
    """Prepare a single prediction request into a scaled numpy array for the model."""
    features = build_feature_frame(input_data)
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    return scaler.transform(features)


if __name__ == "__main__":
    train_and_evaluate_models()
