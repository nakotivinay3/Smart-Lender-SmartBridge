# Smart Lender

## Project Overview
Smart Lender is a machine learning-based loan eligibility prediction system that helps banks and financial institutions decide whether a loan application should be approved or rejected.

## Problem Statement
Manual loan evaluation can be slow, inconsistent and prone to human error. This project uses machine learning to automate the decision-making process using applicant information.

## Objectives
- Read the loan prediction dataset.
- Perform exploratory data analysis.
- Preprocess the data and handle missing values.
- Balance the data using SMOTE.
- Scale features using StandardScaler.
- Train multiple classification models.
- Compare the models and save the best one.
- Deploy the trained model through a Flask web application.

## Real-world Scenarios
- Banks want to reduce manual effort in loan approval.
- Financial institutions need faster and consistent eligibility checks.
- Applicants benefit from quick and transparent outcomes.

## Architecture
The project follows a simple machine learning workflow:
1. Dataset collection
2. Data preprocessing
3. SMOTE balancing
4. Feature scaling
5. Model training and evaluation
6. Flask-based prediction interface

## ER Diagram Explanation
The dataset is tabular and contains applicant characteristics such as income, loan amount, credit history and property area. These features are used to predict the binary target, Loan_Status.

## Workflow
- Load dataset using pandas.
- Explore the data with descriptive statistics and visualization.
- Clean the dataset, encode categories and handle missing values.
- Use SMOTE to balance classes.
- Scale the features.
- Train multiple models and compare results.
- Save the best model and scaler.
- Interact with the model through Flask.

## Dataset Description
The dataset includes the following features:
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status

## Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Flask
- imbalanced-learn

## Software Required
- Python 3.10+
- VS Code or PyCharm
- pip

## Installation Steps
```bash
pip install -r requirements.txt
```

## Execution Steps
```bash
python Dataset/generate_dataset.py
python Training/train_model.py
python app.py
```

## Model Performance
The project compares:
- DecisionTreeClassifier
- RandomForestClassifier
- KNeighborsClassifier
- GradientBoostingClassifier

The best-performing model is saved as model/saved_model.pkl.

## Results
The trained model predicts whether a customer is likely to be approved or rejected for a loan based on the supplied details.

## Conclusion
Smart Lender demonstrates how machine learning can support efficient and automated loan decision-making.

## Future Scope
- Integrate with a real banking dataset.
- Add explainability with SHAP.
- Deploy the application on IBM Cloud or Heroku.
