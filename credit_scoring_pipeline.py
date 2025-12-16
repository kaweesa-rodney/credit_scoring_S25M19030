import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import shap
import matplotlib.pyplot as plt
import seaborn as sns


#load and derive some columns
columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

df = pd.read_csv(
    "data\german.data",
    sep=" ",
    names=columns
)

df.head()

df["credit_risk"] = df["credit_risk"].apply(lambda x: 1 if x == 2 else 0)

df["gender"] = df["personal_status_sex"].apply(
    lambda x: "male" if x in ["A91","A93","A94"] else "female"
)

df["age_group"] = df["age"].apply(lambda x: "young" if x < 25 else "adult")

#train test split
X = df.drop(columns=["credit_risk", "gender", "age_group"])
y = df["credit_risk"]
sensitive = df[["gender", "age_group"]]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive,
    test_size=0.3,
    stratify=y,
    random_state=42
)


#model pipeline
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        solver="liblinear"
    ))
])

print("\n----------Model Training and Prediction--------------")
model.fit(X_train, y_train)

#model evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("F1:", f1_score(y_test, y_pred))

cv_auc = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring="roc_auc"
)
print("CV AUC:", cv_auc.mean())


#fairness
print("\n-------------------------Model Fairness----------------------------------")
results = sens_test.copy()
results["approved"] = (y_prob < 0.4).astype(int)
results["actual"] = y_test.values


di_gender = (
    results.groupby("gender")["approved"].mean()["female"] /
    results.groupby("gender")["approved"].mean()["male"]
)

#default DI gender at 0.4 probability threshold
print(f"\nDisparate Impact(Gender): {di_gender}")

di_age = (
    results.groupby("age_group")["approved"].mean()["young"] /
    results.groupby("age_group")["approved"].mean()["adult"]
)

#default DI age at 0.4 probability threshold
print(f"\nDisparate Impact(Age): {di_age}")


#adjusting probability threshold for age group
results["approved_adjusted"] = np.where(
    (results["age_group"] == "young") & (y_prob < 0.45),
    1,
    results["approved"]
)

approval_by_gender = (
    results.groupby("gender")["approved"]
    .mean()
    .reset_index()
    .rename(columns={"approved": "approval_rate"})
)

print(f"\n{approval_by_gender}")


approval_by_age = (
    results.groupby("age_group")["approved"]
    .mean()
    .reset_index()
    .rename(columns={"approved": "approval_rate"})
)

print(f"\n{approval_by_age}")


#Approval rate by age group after adjusting probability threshold
approval_adj_age = (
    results.groupby("age_group")["approved_adjusted"]
    .mean()
)

print(f"\n{approval_adj_age}")

#DI after adjusted threshold
di_age_adjusted = (
    approval_adj_age["young"] /
    approval_adj_age["adult"]
)

print(f"\n{di_age_adjusted}")

approval_comparison = pd.DataFrame({
    "Age Group": ["Adult", "Young"],
    "Approval Rate (Before)": [
        results.groupby("age_group")["approved"].mean()["adult"],
        results.groupby("age_group")["approved"].mean()["young"]
    ],
    "Approval Rate (After)": [
        approval_adj_age["adult"],
        approval_adj_age["young"]
    ]
})

print(f"\n{approval_comparison}")

y_actual_good = (y_test == 0).astype(int)

print("\n================ PERFORMANCE (Before Mitigation) ================")

print("Accuracy :", accuracy_score(y_actual_good, results["approved"]))
print("Precision:", precision_score(y_actual_good, results["approved"]))
print("Recall   :", recall_score(y_actual_good, results["approved"]))
print("F1 Score :", f1_score(y_actual_good, results["approved"]))


print("\n================ PERFORMANCE (After Mitigation) =================")

print("Accuracy :", accuracy_score(y_actual_good, results["approved_adjusted"]))
print("Precision:", precision_score(y_actual_good, results["approved_adjusted"]))
print("Recall   :", recall_score(y_actual_good, results["approved_adjusted"]))
print("F1 Score :", f1_score(y_actual_good, results["approved_adjusted"]))


#model explanation(shap)
print("\n---------------------SHAP Explainability------------------------")
X_train_transformed = model.named_steps["prep"].transform(X_train)
X_test_transformed  = model.named_steps["prep"].transform(X_test)

feature_names = model.named_steps["prep"].get_feature_names_out()


explainer = shap.LinearExplainer(
    model.named_steps["clf"],
    X_train_transformed
)

shap_values = explainer.shap_values(X_test_transformed)


# Summary Plot with wording
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

#plt.show()


#SHAP importance
shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
}).sort_values(by="Mean |SHAP Value|", ascending=False)

# interpretations
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

print("\nTop SHAP Feature Contributions:")
print(shap_importance.head(10))


# explanation of output
print("\nModel Explanation Summary:")
for _, row in shap_importance.head(20).iterrows():
    explanation = row["Interpretation"]
    if pd.notna(explanation):
        print(f"- {explanation}")




#Error Analysis
print("\n--------------------Error Analysis---------------")
errors = X_test.copy()
errors["actual"] = y_test
errors["pred"] = y_pred

false_positives = errors[(errors["actual"] == 1) & (errors["pred"] == 0)]
false_negatives = errors[(errors["actual"] == 0) & (errors["pred"] == 1)]

print("\n================ ERROR ANALYSIS =================")

print(f"False Positives: {len(false_positives)}")
print("Interpretation: Applicants predicted as low-risk but who actually defaulted.")
print("Business impact: Potential financial loss due to loan defaults.\n")

print(f"False Negatives: {len(false_negatives)}")
print("Interpretation: Creditworthy applicants incorrectly rejected.")
print("Business impact: Missed revenue and potential customer dissatisfaction.")

