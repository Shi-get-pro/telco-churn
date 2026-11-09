
## Telco Customer Churn – Classification Models & Streamlit App
## 1. Problem Statement
A telecom company loses revenue every time a customer leaves (churns). The goal of this project is to predict, from a customer's account, service and billing details, whether that customer will churn (Yes / No).
Five classification models are trained and compared on the same held-out test set using six evaluation metrics. The trained models are then served through an interactive Streamlit web app where a user can upload test data, pick a model and see how it performs.
________________________________________
## 2. Dataset Description
Item	Detail
Source	IBM Telco Customer Churn dataset (Telco-Customer-Churn.csv, publicly available on Kaggle)
Rows × Columns	7,043 × 21 (7,032 × 20 after cleaning)
Target	Churn – Yes / No
Class balance	No: 5,174 (73.5%) · Yes: 1,869 (26.5%) → imbalanced
## Features (19 inputs):
•	Numeric (3): tenure, MonthlyCharges, TotalCharges
•	Categorical (16): gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
## Data cleaning
1.	Dropped customerID – it is only an identifier, not a predictive feature.
2.	Converted TotalCharges from text to numeric. 11 rows had a blank value; all of them had tenure = 0 (brand-new customers), so they were dropped.
3.	Encoded the target as Yes → 1, No → 0.
Train / test split: 80 / 20, stratified on Churn, random_state = 42 → Train: 5,625 rows · Test: 1,407 rows (churn rate 26.6% in both).
## Preprocessing (inside a scikit-learn Pipeline, so it is saved together with each model):
•	StandardScaler on numeric columns
•	OneHotEncoder(handle_unknown="ignore") on categorical columns
Because the classes are imbalanced, models are judged on AUC, Recall, F1 and MCC, not on accuracy alone.
________________________________________
## 3. Models Used
#	Model	Key settings
1	Logistic Regression	max_iter=1000
2	Decision Tree	max_depth=5, min_samples_leaf=20
3	K-Nearest Neighbors (KNN)	n_neighbors=5
4	Naive Bayes	GaussianNB
5	Support Vector Machine (SVM)	RBF kernel, C=1.0, probability=True

## Comparison of evaluation metrics (held-out test set, n = 1,407)

<img width="591" height="145" alt="image" src="https://github.com/user-attachments/assets/54b1c21b-90cb-4d7f-a5c6-6a3603e54840" />

## Observations on model performance
ML Model Name	Observation
Logistic Regression	Best overall model: highest Accuracy, AUC, Precision, F1 and MCC. It gives the best balance between catching churners and avoiding false alarms, and it is easy to interpret.

Decision Tree	Consistently in the middle and very close to Logistic Regression on F1 (0.6093). Its recall (0.6150) is higher than LR's, and its rules are easy to explain, so it is a strong alternative.

KNN	Weakest overall: lowest AUC (0.7818) and lowest MCC (0.4028). Distance-based voting struggles with the many one-hot encoded categorical features.

Naive Bayes	Highest recall (0.8289), so it catches the most churners, but it has the lowest precision (0.4493) and accuracy (0.6844), meaning many false alarms. Useful only when missing a churner is far more costly than a wasted retention offer.

Support Vector Machine	Most conservative model: good precision (0.6458) but the lowest recall (0.4973) and F1 (0.5619). It misses about half of the actual churners at the default threshold.

<img width="2179" height="404" alt="assignment2" src="https://github.com/user-attachments/assets/09cbdc34-2ed3-4a91-bde5-9cbb99d1aab4" />

## Conclusion: Logistic Regression is the recommended model. For Logistic Regression, the classification report shows the "No churn" class at precision 0.85 / recall 0.89 and the "Churn" class at precision 0.65 / recall 0.57.
## Tech Stack: Python · pandas · NumPy · scikit-learn · matplotlib · joblib · Streamlit




