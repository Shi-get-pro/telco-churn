from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).parent
# Models may sit in a model/ folder or next to app.py
MODEL_DIR = APP_DIR / "model" if (APP_DIR / "model").is_dir() else APP_DIR
TARGET = "Churn"
# File names exactly as saved by the notebook
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")
st.title("Telco Customer Churn - Model Evaluation")
st.caption(
    "Upload the held-out test CSV, pick a model, and see how the saved model performs. "
    "No retraining happens here."
)


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, scores


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(
        "The CSV needs every input feature plus the `Churn` column (Yes/No). "
        "Use `test_data.csv` from the repo."
    )

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin.")
    st.stop()

# ---------------- Data ----------------
data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error("The uploaded file has no 'Churn' column.")
    st.stop()

data = data.drop(columns=["customerID"], errors="ignore")
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna().reset_index(drop=True)

y_true = (data[TARGET].astype(str).str.strip() == "Yes").astype(int)
X = data.drop(columns=[TARGET])

st.subheader("Uploaded data")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Churners", int(y_true.sum()))
c3.metric("Churn rate", f"{y_true.mean():.1%}")
with st.expander("Preview first 10 rows"):
    st.dataframe(data.head(10))

if y_true.nunique() < 2:
    st.error("The Churn column needs both 'Yes' and 'No' values to compute AUC and MCC.")
    st.stop()

tab_single, tab_all = st.tabs(["Selected model", "Compare all models"])

# ---------------- Selected model ----------------
with tab_single:
    try:
        model = load_model(model_name)
        pred, scores = evaluate(model, X, y_true)
    except Exception as e:
        st.error(f"Could not score this file with {model_name}: {e}")
        st.stop()

    st.subheader(f"Evaluation metrics - {model_name}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, scores.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true, pred, target_names=["No churn", "Churn"], output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

# ---------------- All models ----------------
with tab_all:
    st.subheader("All five models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        try:
            _, s = evaluate(load_model(name), X, y_true)
            rows.append({"ML Model Name": name, **s})
        except Exception as e:
            st.warning(f"Skipped {name}: {e}")
    if rows:
        table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
        st.dataframe(table)
        st.bar_chart(table[["Accuracy", "AUC", "F1", "MCC"]])