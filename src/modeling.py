"""
Machine learning modeling module for diabetes readmission prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from xgboost import XGBClassifier


class DiabetesReadmissionModel:
    """Model for predicting 30-day readmission for diabetes patients."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.is_trained = False
        self.feature_names = None

    def _get_model(self, model_type: str):

        if model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )

        elif model_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using probability + threshold.
        Default threshold = 0.5 (safe for Predictions page)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        y_prob = self.model.predict_proba(X)[:, 1]
        return (y_prob >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            return None

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


# ------------------------- DATA SPLITTING -------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str = "readmitted_30_days",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


# ------------------------- EVALUATION (THRESHOLD ENABLED) -------------------------

def evaluate_model(
    model: DiabetesReadmissionModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }
    }


# ------------------------- MODEL COMPARISON -------------------------
# ------------------------- MODEL COMPARISON -------------------------

def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    xgb_threshold: float = 0.5
):
    results = []

    model_settings = [
        ("logistic_regression", 0.5),
        ("random_forest", 0.5),
        ("xgboost", xgb_threshold)
    ]

    for model_type, threshold in model_settings:
        model = DiabetesReadmissionModel(model_type)
        model.train(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            threshold=threshold
        )

        results.append({
            "Model": model_type.replace("_", " ").title(),
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
            "ROC AUC": metrics["roc_auc"],
            "Threshold Used": threshold
        })

    return pd.DataFrame(results)

# ------------------------- FEATURE IMPORTANCE -------------------------

def get_top_predictors(
    model: DiabetesReadmissionModel,
    top_n: int = 20
):
    if not model.is_trained:
        raise ValueError("Model must be trained before extracting feature importance")

    importance_df = model.get_feature_importance()

    if importance_df is None:
        return None

    return importance_df.head(top_n)
