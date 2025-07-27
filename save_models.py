import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load and preprocess the data
df = pd.read_csv("loan_eligibility_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert Loan_Eligibility to binary
df["Loan_Eligibility"] = df["Loan_Eligibility"].map({"Eligible": 1, "Not Eligible": 0})

# Features and targets
X = df.drop(columns=["Loan_Eligibility", "Interest_Rate", "Loan_Tenure"])
y1 = df["Loan_Eligibility"]
y2 = df["Interest_Rate"]
y3 = df["Loan_Tenure"]

# Add derived features
X["Debt-to-Income Ratio"] = X["Total_Debt"] / X["Income"]
X["Loan-to-Value Ratio"] = X["Loan_Amount"] / X["Total_Assets"]
X["Employment_Stability_Score"] = X["Experience"] * X["Salary_Growth"]

# Label encode categorical columns
categorical_columns = [
    "Employment_Stability", "Employment_Type", "Loan_Type", "Purpose", 
    "Education", "Collateral", "Insurance", "Tax_History", 
    "Residence_Type", "Residence"
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)
_, _, y3_train, y3_test = train_test_split(X, y3, test_size=0.2, random_state=42)

# Train models
eligibility_model = GaussianNB()
interest_model = BayesianRidge()
tenure_model = BayesianRidge()

eligibility_model.fit(X_train, y1_train)
interest_model.fit(X_train, y2_train)
tenure_model.fit(X_train, y3_train)

# Save models
joblib.dump(eligibility_model, 'models/eligibility_model.joblib')
joblib.dump(interest_model, 'models/interest_model.joblib')
joblib.dump(tenure_model, 'models/tenure_model.joblib')

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.joblib')

print("Models and encoders have been saved successfully!") 