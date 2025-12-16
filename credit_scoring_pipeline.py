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
df = pd.read_csv("data/german_credit.csv")

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

