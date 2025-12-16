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
    page_title="Fairness-Aware Credit Scoring",
    layout="wide"
)

st.title("Fairness-Aware Credit Scoring System")

st.markdown("""
This application demonstrates an end-to-end **credit scoring system**
with **fairness auditing, mitigation, explainability, and applicant-level scoring**.
""")

# -------------------------------------------------
# Load data
# -------------------------------------------------
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
        lambda x: "male" if x in ["A91","A93","A94"] else "female"
    )
    df["age_group"] = df["age"].apply(lambda x: "young" if x < 25 else "adult")

    return df

df = load_data()

# -------------------------------------------------
# Sidebar – decision policy
# -------------------------------------------------
st.sidebar.header("Decision Policy")

base_threshold = st.sidebar.slider(
    "Base Approval Threshold", 0.30, 0.60, 0.40, 0.01
)

young_threshold = st.sidebar.slider(
    "Young Applicant Threshold", 0.30, 0.60, 0.45, 0.01
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
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Credit Risk Prediction",
    "Fairness Auditing & Mitigation",
    "Explainability (SHAP)",
    "In-App Risk Scoring"
])


# PREDICTION

with tab1:
    st.subheader("Model Performance")

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


# FAIRNESS
with tab2:
    st.subheader("Fairness Metrics")

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

    st.markdown("""
    **80% Rule**
    - DI ≥ 0.80 indicates acceptable fairness
    - Age-based predict pob. threshold improves age fairness, this is what we adjust
    """)


# EXPLAINABILITY (SHAP)
with tab3:
    st.subheader("🔍 SHAP Explainability")

    st.markdown("""
    This section explains **why** the model makes its predictions.
    - Positive SHAP values increase default risk
    - Negative SHAP values reduce default risk
    """)

    # -------------------------------------------------
    # Prepare transformed data
    # -------------------------------------------------
    X_train_transformed = model.named_steps["prep"].transform(X_train)
    X_test_transformed  = model.named_steps["prep"].transform(X_test)

    feature_names = model.named_steps["prep"].get_feature_names_out()

    # -------------------------------------------------
    # SHAP explainer
    # -------------------------------------------------
    explainer = shap.LinearExplainer(
        model.named_steps["clf"],
        X_train_transformed
    )

    shap_values = explainer.shap_values(X_test_transformed)

    # -------------------------------------------------
    # SHAP summary plot (with wording)
    # -------------------------------------------------
    fig, ax = plt.subplots()
    plt.title(
        "SHAP Summary Plot: Feature Impact on Loan Default Risk\n"
        "Positive values increase default risk; negative values reduce risk",
        fontsize=11
    )

    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )

    st.pyplot(fig)

    # -------------------------------------------------
    # SHAP importance table
    # -------------------------------------------------
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean |SHAP Value|", ascending=False)

    interpretation_map = {
        "num__credit_amount": "Higher loan amounts increase default risk",
        "num__duration": "Longer loan durations increase default risk",
        "num__age": "Older applicants tend to have lower default risk",
        "cat__employment_A75": "Stable employment reduces default risk",
        "cat__savings_A65": "Higher savings reduce default risk"
    }

    shap_importance["Interpretation"] = shap_importance["Feature"].map(
        interpretation_map
    )

    st.subheader("Top SHAP Feature Contributions")
    st.dataframe(
        shap_importance.head(10),
        use_container_width=True
    )

    # -------------------------------------------------
    # Human-readable explanation summary
    # -------------------------------------------------
    st.subheader("Summary")

    for _, row in shap_importance.head(20).iterrows():
        if pd.notna(row["Interpretation"]):
            st.markdown(f"- **{row['Interpretation']}**")


#IN-APP RISK SCORING
with tab4:
    st.subheader("Credit Scoring")

    with st.form("applicant_form"):
        applicant = {}
        for col in X.columns:
            if col in cat_cols:
                applicant[col] = st.selectbox(col, sorted(df[col].unique()))
            else:
                applicant[col] = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max())
                )

        submitted = st.form_submit_button("Score Applicant")

    if submitted:
        applicant_df = pd.DataFrame([applicant])

        prob = model.predict_proba(applicant_df)[0, 1]

        age_group = "young" if applicant["age"] < 25 else "adult"
        threshold = young_threshold if age_group == "young" else base_threshold

        decision = "APPROVED" if prob < threshold else "REJECTED"

        st.success(f"Predicted Default Probability: {round(prob, 3)}")
        st.info(f"Threshold Applied: {threshold}")
        st.warning(f"Final Decision: {decision}")