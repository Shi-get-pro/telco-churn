# Telco Customer Churn - Classification Models and Streamlit App

## a.  Problem Statement 
  Predict whether a telecom customer will churn (leave the company) using their account, service and Billing details. 
  This is a Binary classification Problem. Five Classifications are trained on the same dataset and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. 
  The trained models are served through a Streamlit app.

  ## b. Dataset Description
  Source - IBM Telco Customer Churn Dataset from Kaggle, 
  7043 records with 11 blanks in TotalCharges, After removing 11 records, we have 7032 records. Those 11 records are new customers and tenure was 0. New customers are not billed yet. 
  Features in the data set = 19 Input features after dropping CustomerID which is Unique Identifier.
  Numeric : Tenure, MonthlyCharges, TotalCharges
  Categorical/ Binary : Gender, Sr. Citizen, Partner, Dependants, PhoneServices, Multiplelines, Internetservices, Online Security, Online Backup, Dvice protection, TechSupport, StreamingTV, StreamingMovies, Contract, Paperlessbilling, PaymentMethod.
  Target = Churn (Yes = 1, No = 0 ), Class Balance is about 73/5% No and 26.5% Yes, so the classes are imbalanced. 
  Preprocessing : Numeric columns standardized (StandardScaler), Categorical Columns one-hot encoded, both are inside a scikit-learn Pipeline saved with each model. 
  Split: Stratified 80/20 train/test split with randowm_state = ___. Train = ___ rows, test = ___ rows (Saved as test_data.csv) 

  Git hub repo link - : 
