# Telco Customer Churn – Classification Models & Streamlit App

**Student:** Mary Shirly S  **ID:** 2026AK05393  **Subject:** Machine Learning – Assignment 2

🔗 **Live app:** `<add your Streamlit Cloud link here>`

---

## 1. Problem Statement

A telecom company loses revenue every time a customer leaves (churns). The goal of this project is to predict, from a customer's account, service and billing details, **whether that customer will churn (Yes / No)**.

Five classification models are trained and compared on the same held-out test set using six evaluation metrics. The trained models are then served through an interactive **Streamlit** web app where a user can upload test data, pick a model and see how it performs.

---

## 2. Dataset Description

| Item | Detail |
|---|---|
| Source | IBM Telco Customer Churn dataset (`Telco-Customer-Churn.csv`, publicly available on Kaggle) |
| Rows × Columns | 7,043 × 21 (7,032 × 20 after cleaning) |
| Target | `Churn` – Yes / No |
| Class balance | No: 5,174 (73.5%) · Yes: 1,869 (26.5%) → **imbalanced** |

**Features (19 inputs):**

- **Numeric (3):** `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categorical (16):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

**Data cleaning**

1. Dropped `customerID` – it is only an identifier, not a predictive feature.
2. Converted `TotalCharges` from text to numeric. 11 rows had a blank value; all of them had `tenure = 0` (brand-new customers), so they were dropped.
3. Encoded the target as `Yes → 1`, `No → 0`.

**Train / test split:** 80 / 20, stratified on `Churn`, `random_state = 42`
→ Train: 5,625 rows · Test: 1,407 rows (churn rate 26.6% in both).

**Preprocessing** (inside a scikit-learn `Pipeline`, so it is saved together with each model):

- `StandardScaler` on numeric columns
- `OneHotEncoder(handle_unknown="ignore")` on categorical columns

Because the classes are imbalanced, models are judged on **AUC, Recall, F1 and MCC**, not on accuracy alone.

---

## 3. Models Used

| # | Model | Key settings |
|---|---|---|
| 1 | Logistic Regression | `max_iter=1000` |
| 2 | Decision Tree | `max_depth=5`, `min_samples_leaf=20` |
| 3 | K-Nearest Neighbors (KNN) | `n_neighbors=5` |
| 4 | Naive Bayes | `GaussianNB` |
| 5 | Support Vector Machine (SVM) | RBF kernel, `C=1.0`, `probability=True` |

### Comparison of evaluation metrics (held-out test set, n = 1,407)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8045** | **0.8359** | **0.6495** | 0.5749 | **0.6099** | **0.4818** |
| Decision Tree | 0.7903 | 0.8264 | 0.6037 | 0.6150 | 0.6093 | 0.4661 |
| KNN | 0.7605 | 0.7818 | 0.5457 | 0.5909 | 0.5674 | 0.4028 |
| Naive Bayes | 0.6844 | 0.8040 | 0.4493 | **0.8289** | 0.5827 | 0.4074 |
| Support Vector Machine | 0.7939 | 0.7842 | 0.6458 | 0.4973 | 0.5619 | 0.4364 |

*Bold = best value in that column.*

### Observations on model performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Best overall model: highest Accuracy, AUC, Precision, F1 and MCC. It gives the best balance between catching churners and avoiding false alarms, and it is easy to interpret. |
| Decision Tree | Consistently in the middle and very close to Logistic Regression on F1 (0.6093). Its recall (0.6150) is higher than LR's, and its rules are easy to explain, so it is a strong alternative. |
| KNN | Weakest overall: lowest AUC (0.7818) and lowest MCC (0.4028). Distance-based voting struggles with the many one-hot encoded categorical features. |
| Naive Bayes | Highest recall (0.8289), so it catches the most churners, but it has the lowest precision (0.4493) and accuracy (0.6844), meaning many false alarms. Useful only when missing a churner is far more costly than a wasted retention offer. |
| Support Vector Machine | Most conservative model: good precision (0.6458) but the lowest recall (0.4973) and F1 (0.5619). It misses about half of the actual churners at the default threshold. |

**Conclusion:** **Logistic Regression is the recommended model.** For Logistic Regression, the classification report shows the "No churn" class at precision 0.85 / recall 0.89 and the "Churn" class at precision 0.65 / recall 0.57.

---

## 4. Streamlit App

The app (`app.py`) **only evaluates the saved models. No retraining happens in the app.**

**Features**

- 📤 Upload a test CSV (use `test_data.csv` from this repo)
- 🔽 Pick any of the five saved models from a dropdown
- 📊 Summary of the uploaded data: rows, number of churners, churn rate and a preview of the first rows
- 📈 All six evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- 🟦 Confusion matrix and classification report for the selected model
- 🏁 Side-by-side comparison of all five models on the uploaded data, with a bar chart

**Required input format:** the CSV must contain all 19 input feature columns listed above plus a `Churn` column with `Yes` / `No` values. A `customerID` column, if present, is ignored.

---

## 5. Repository Structure

```
├── app.py                                  # Streamlit app
├── requirements.txt                        # Python dependencies
├── AI_ML_Assignment2_2026AK05393q.ipynb    # Training & evaluation notebook
├── test_data.csv                           # Held-out test set (1,407 rows) for the app
├── metric.csv                              # Metrics table from the notebook
├── logistic_regression.joblib              # Saved model pipelines
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
├── support_vector_machine.joblib
└── README.md
```

---


## 8. Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · joblib · Streamlit
