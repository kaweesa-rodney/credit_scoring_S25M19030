import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring & Fairness Dashboard",
    layout="wide"
)

st.title("Fairness-Aware Credit Scoring System")

st.markdown("""
This dashboard demonstrates:
- Credit risk prediction
- Fairness auditing & mitigation
- Explainability (SHAP)
- In-app risk scoring
- Downloadable audit reports
""")

# Load data
@st.cache_data
def load_data():
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "credit_risk"
    ]

    df = pd.read_csv("data/german.data", sep=" ", names=columns)

    df["credit_risk"] = df["credit_risk"].apply(lambda x: 1 if x == 2 else 0)
    df["gender"] = df["personal_status_sex"].apply(
        lambda x: "male" if x in ["A91", "A93", "A94"] else "female"
    )
    df["age_group"] = df["age"].apply(lambda x: "young" if x < 25 else "adult")

    return df

df = load_data()


# Sidebar – decision policy
st.sidebar.header("Decision Policy")

base_threshold = st.sidebar.slider(
    "Base Approval Threshold",
    0.30, 0.60, 0.40, 0.01
)

young_threshold = st.sidebar.slider(
    "Young Applicant Threshold",
    0.30, 0.60, 0.45, 0.01
)

# -------------------------------------------------
# Train model
# -------------------------------------------------
X = df.drop(columns=["credit_risk", "gender", "age_group"])
y = df["credit_risk"]
sensitive = df[["gender", "age_group"]]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        solver="liblinear"
    ))
])

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.3, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Model performance
# -------------------------------------------------
st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", round(accuracy_score(y_test, model.predict(X_test)), 3))
col2.metric("ROC-AUC", round(roc_auc_score(y_test, y_prob), 3))
col3.metric(
    "Precision",
    round(precision_score((y_test == 0), y_prob < base_threshold), 3)
)
col4.metric(
    "Recall",
    round(recall_score((y_test == 0), y_prob < base_threshold), 3)
)


# Fairness analysis
st.header("Fairness Analysis")

results = sens_test.copy()
results["approved"] = (y_prob < base_threshold).astype(int)
results["actual"] = y_test.values

results["approved_adjusted"] = np.where(
    (results["age_group"] == "young") & (y_prob < young_threshold),
    1,
    results["approved"]
)

di_gender = (
    results.groupby("gender")["approved_adjusted"].mean()["female"] /
    results.groupby("gender")["approved_adjusted"].mean()["male"]
)

di_age = (
    results.groupby("age_group")["approved_adjusted"].mean()["young"] /
    results.groupby("age_group")["approved_adjusted"].mean()["adult"]
)

st.write("**Disparate Impact (Gender):**", round(di_gender, 3))
st.write("**Disparate Impact (Age):**", round(di_age, 3))




# risk scoring
st.header("Applicant-Level Credit Scoring")

with st.form("applicant_form"):
    applicant = {}
    for col in X.columns:
        if col in cat_cols:
            applicant[col] = st.selectbox(col, sorted(df[col].unique()))
        else:
            applicant[col] = st.number_input(
                col, float(df[col].min()), float(df[col].max())
            )

    submitted = st.form_submit_button("Score Applicant")

if submitted:
    applicant_df = pd.DataFrame([applicant])

    prob = model.predict_proba(applicant_df)[0, 1]

    age_group = "young" if applicant["age"] < 25 else "adult"
    threshold = young_threshold if age_group == "young" else base_threshold

    decision = "APPROVED" if prob < threshold else "REJECTED"

    st.success(f"Predicted Default Probability: {round(prob, 3)}")
    st.info(f"Decision Threshold Used: {threshold}")
    st.warning(f"Final Decision: {decision}")



# SHAP explainability
st.header("🔍 Model Explainability (SHAP)")

X_train_t = model.named_steps["prep"].transform(X_train)
X_test_t = model.named_steps["prep"].transform(X_test)
feature_names = model.named_steps["prep"].get_feature_names_out()

explainer = shap.LinearExplainer(model.named_steps["clf"], X_train_t)
shap_values = explainer.shap_values(X_test_t)

fig, ax = plt.subplots()
shap.summary_plot(
    shap_values,
    X_test_t,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig)


# Error analysis
st.header("Error Analysis")

errors = X_test.copy()
errors["actual"] = y_test
errors["pred"] = (y_prob < base_threshold).astype(int)

fp = len(errors[(errors["actual"] == 1) & (errors["pred"] == 0)])
fn = len(errors[(errors["actual"] == 0) & (errors["pred"] == 1)])

st.write(f"**False Positives:** {fp} → Potential financial loss")
st.write(f"**False Negatives:** {fn} → Missed revenue opportunities")