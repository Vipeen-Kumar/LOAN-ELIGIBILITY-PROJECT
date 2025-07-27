# LOAN-ELIGIBILITY-PROJECT

## Project Overview
The Loan Eligibility Project is a machine learning-based application designed to predict loan eligibility for applicants. It uses a dataset of loan applications to train models that can determine eligibility, interest rates, and tenure for loans. The project includes pre-trained models and scripts for making predictions and saving results.

## Features
- Predict loan eligibility based on applicant details.
- Predict interest rates and loan tenure.
- Pre-trained models for quick predictions.
- Easy-to-use Python scripts and Jupyter Notebook for exploration.

## Project Structure
- `app.py`: Main script for running the application.
- `loan_eligibility_dataset.csv`: Dataset used for training and testing.
- `Loan_Eligibility_Model.ipynb`: Jupyter Notebook for model training and analysis.
- `save_models.py`: Script to save trained models.
- `models/`: Directory containing pre-trained models (`eligibility_model.joblib`, `interest_model.joblib`, `label_encoders.joblib`, `tenure_model.joblib`).

## Prerequisites
- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Steps to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vipeen-Kumar/LOAN-ELIGIBILITY-PROJECT.git
   cd LOAN-ELIGIBILITY-PROJECT
   ```

2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Execute the main script to start the application:
   ```bash
   python app.py
   ```

4. **Explore the Jupyter Notebook** (Optional):
   Open `Loan_Eligibility_Model.ipynb` in Jupyter Notebook to explore the model training and analysis.

## Notes
- Ensure that the `models/` directory contains the pre-trained models before running the application.
- Modify the dataset or retrain the models as needed using the provided notebook.