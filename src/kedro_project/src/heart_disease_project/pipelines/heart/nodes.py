# src/<pkg>/pipelines/heart/nodes.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------- Split ----------
def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop("target", axis=1)
    y = df["target"].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ---------- Preprocessor ----------
def build_preprocessor(numeric_features, binary_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

# ---------- Train RF (with optional GridSearch) ----------
def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    rf_params: dict,
):
    clf = RandomForestClassifier(**rf_params.get("base_params", {}))
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    best_params = rf_params.get("base_params", {})

    if rf_params.get("gridsearch", False):
        param_grid = rf_params["param_grid"]
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        model = pipe.fit(X_train, y_train)

    return model, best_params

# ---------- Evaluate ----------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    # Some preprocessors may drop predict_proba (rare); guard it:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_prob))
    else:
        roc = None

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
