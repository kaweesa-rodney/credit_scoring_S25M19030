import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

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

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive,
    test_size=0.3,
    stratify=y,
    random_state=42
)


#model pipeline
pipeline = Pipeline(steps=[
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), X.select_dtypes(exclude="object").columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         X.select_dtypes(include="object").columns)
    ])),
    ("clf", LogisticRegression(class_weight="balanced", solver="liblinear"))
])


#hyperparameter tuning
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=5
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

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

