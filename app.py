import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

# Check if models exist
if not os.path.exists('models/eligibility_model.joblib'):
    st.error("""
    Models not found! Please run the following commands first:
    ```
    python save_models.py
    ```
    """)
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .eligible {
        background-color: #d4edda;
        color: #155724;
    }
    .not-eligible {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏦 Loan Eligibility Predictor")
st.markdown("""
This application helps predict loan eligibility based on various factors. 
Fill in your information below to get an instant assessment.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (₹)", min_value=0, value=50000)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=5000)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
    experience = st.number_input("Work Experience (Years)", min_value=0, max_value=50, value=5)
    
    employment_stability = st.selectbox(
        "Employment Stability",
        ["Permanent", "Contract Based", "New Joining"]
    )
    
    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-Employed", "Business Owner"]
    )
    
    education = st.selectbox(
        "Education",
        ["High School", "Bachelor's", "Master's", "PhD"]
    )

with col2:
    st.subheader("Loan Information")
    loan_type = st.selectbox(
        "Loan Type",
        ["Home Loan", "Personal Loan", "Car Loan", "Education Loan"]
    )
    
    purpose = st.selectbox(
        "Loan Purpose",
        ["House Purchase", "Car Purchase", "Education", "Business Expansion", "Medical Emergency"]
    )
    
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
    total_debt = st.number_input("Total Existing Debt (₹)", min_value=0, value=0)
    salary_growth = st.slider("Annual Salary Growth (%)", min_value=0, max_value=20, value=5)
    total_assets = st.number_input("Total Assets Value (₹)", min_value=0, value=200000)
    
    collateral = st.selectbox("Collateral Available", ["Yes", "No"])
    insurance = st.selectbox("Insurance", ["Yes", "No"])
    tax_history = st.selectbox("Tax History", ["Good", "Average", "Poor"])
    residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Company Provided"])
    residence = st.selectbox("Residence Area", ["Urban", "Semi-Urban", "Rural"])

# Create a dictionary of the input data
def prepare_input_data():
    input_data = {
        "Age": age,
        "Income": income,
        "Expenses": expenses,
        "CIBIL_Score": cibil_score,
        "Experience": experience,
        "Employment_Stability": employment_stability,
        "Employment_Type": employment_type,
        "Loan_Type": loan_type,
        "Purpose": purpose,
        "Education": education,
        "Loan_Amount": loan_amount,
        "Collateral": collateral,
        "Insurance": insurance,
        "Tax_History": tax_history,
        "Residence_Type": residence_type,
        "Residence": residence,
        "Total_Debt": total_debt,
        "Salary_Growth": salary_growth,
        "Total_Assets": total_assets
    }
    return pd.DataFrame([input_data])

# Add derived features
def add_derived_features(df):
    df = df.copy()
    df["Debt-to-Income Ratio"] = df["Total_Debt"] / df["Income"]
    df["Loan-to-Value Ratio"] = df["Loan_Amount"] / df["Total_Assets"]
    df["Employment_Stability_Score"] = df["Experience"] * df["Salary_Growth"]
    return df

# Predict button
if st.button("Predict Loan Eligibility", type="primary"):
    try:
        # Prepare input data
        input_df = prepare_input_data()
        
        # Load label encoders
        try:
            label_encoders = joblib.load('models/label_encoders.joblib')
        except FileNotFoundError:
            st.error("Label encoders not found. Please run 'python save_models.py' first.")
            st.stop()
            
        # Apply label encoding to categorical columns
        categorical_columns = [
            "Employment_Stability", "Employment_Type", "Loan_Type", "Purpose", 
            "Education", "Collateral", "Insurance", "Tax_History", 
            "Residence_Type", "Residence"
        ]
        
        for col in categorical_columns:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Add derived features
        input_df = add_derived_features(input_df)
        
        # Load the models
        try:
            eligibility_model = joblib.load('models/eligibility_model.joblib')
            interest_model = joblib.load('models/interest_model.joblib')
            tenure_model = joblib.load('models/tenure_model.joblib')
        except FileNotFoundError as e:
            st.error(f"Model files not found. Please run 'python save_models.py' first.")
            st.stop()
        
        # Make predictions
        eligibility = eligibility_model.predict(input_df)[0]
        interest_rate = interest_model.predict(input_df)[0]
        tenure = tenure_model.predict(input_df)[0]
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        # Eligibility result
        if eligibility == 1:
            st.markdown("""
            <div class="prediction-box eligible">
                <h3>✅ Congratulations! You are eligible for the loan.</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box not-eligible">
                <h3>❌ Sorry, you are not eligible for the loan at this time.</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Display additional predictions
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Interest Rate", f"{interest_rate:.2f}%")
        with col2:
            st.metric("Recommended Loan Tenure", f"{tenure:.1f} years")
        
        # Display risk factors and recommendations
        st.subheader("📋 Analysis")
        
        # Risk factors based on input values
        risk_factors = []
        if cibil_score < 700:
            risk_factors.append("Low CIBIL Score")
        if total_debt/income > 0.5:
            risk_factors.append("High Debt-to-Income Ratio")
        if experience < 2:
            risk_factors.append("Limited Work Experience")
        
        if risk_factors:
            st.warning("Risk Factors: " + ", ".join(risk_factors))
        
        # Recommendations
        st.info("""
        💡 Recommendations:
        - Maintain a good credit score by paying bills on time
        - Keep your debt-to-income ratio low
        - Ensure stable employment history
        - Build a strong savings record
        """)
        
    except Exception as e:
        st.error(f"""
        An error occurred: {str(e)}
        
        Please make sure you have:
        1. Run 'python save_models.py' to train and save the models
        2. Have the required dataset 'loan_eligibility_dataset.csv' in the correct location
        3. Have all required dependencies installed
        """)
        


# table :- 


# Data for the loan eligibility table
data = {
    "Feature": [
        "Age", "Income", "Expenses", "CIBIL Score", "Experience", 
        "Employment Stability", "Employment Type", "Loan Type", "Purpose", 
        "Education", "Loan Amount", "Loan Tenure", "Collateral", "Insurance", 
        "Tax History", "Residence Type", "Interest Rate", "Residence", 
        "Total Debt", "Salary Growth", "Total Assets"
    ],
    "Minimum Expected Value": [
        "21 years", "₹25,000/month (₹3,00,000/year)", "< 50% of income", "700+", 
        "1 year", "Stable", "Salaried or Self-Employed", "Any", "Any", 
        "Graduate or higher", "< 50% of Total Assets", "5+ years", 
        "Yes (if loan > ₹5L)", "Yes (preferred)", "Good", "Owned (preferred)", 
        "≤ 12%", "Urban or Semi-Urban", "< 40% of income", "≥ 5% annually", 
        "₹5,00,000"
    ],
    "Reason": [
        "Banks typically require applicants to be at least 21", 
        "Stable income is essential for loan repayment", 
        "Expenses should not exceed 50% of monthly income", 
        "A good credit score increases approval chances", 
        "At least one year of job experience is required", 
        "Stable employment ensures regular income", 
        "Salaried individuals have a higher approval rate", 
        "Loan type does not affect eligibility directly", 
        "Purpose does not directly impact eligibility", 
        "Higher education increases job stability", 
        "A loan amount lower than assets is safer", 
        "Longer tenure reduces EMI burden", 
        "Collateral is needed for large loans", 
        "Having insurance reduces risk", 
        "Clean tax history improves eligibility", 
        "Owned residence adds financial stability", 
        "Lower interest rates are preferred", 
        "Rural applicants may face stricter conditions", 
        "Debt should not exceed 40% of monthly income", 
        "Growth ensures long-term financial stability", 
        "Having assets increases approval chances"
    ]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Streamlit App
st.title("Loan Eligibility Criteria")
st.table(df)





# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ by VIPEEN KUMAR and Prathmesh Thanekar</p>
</div>
""", unsafe_allow_html=True) 