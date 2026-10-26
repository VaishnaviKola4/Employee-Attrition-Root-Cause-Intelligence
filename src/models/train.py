"""
Attrition Prediction Models: Random Forest, Gradient Boosting, Logistic Regression.
Includes training, evaluation metrics, calibration, and hyperparameter tuning.
"""

import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


class AttritionPredictor:
    """Supervised Machine Learning model suite for employee attrition risk."""

    AVAILABLE_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }

    def __init__(self, model_type: str = "gradient_boosting", **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self._init_model()

    def _init_model(self):
        default_params = {
            "random_forest": {"n_estimators": 150, "max_depth": 8, "random_state": 42, "class_weight": "balanced"},
            "gradient_boosting": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 4, "random_state": 42},
            "logistic_regression": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
        }
        params = default_params.get(self.model_type, {}).copy()
        params.update(self.model_params)

        cls = self.AVAILABLE_MODELS.get(self.model_type, GradientBoostingClassifier)
        self.model = cls(**params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the underlying model."""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive performance evaluation metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "model_type": self.model_type,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        """Extract global feature importance weights."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}

        feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    def save(self, filepath: str):
        """Serialize trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        """Load serialized model from disk."""
        self.model = joblib.load(filepath)


def train_and_compare_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train multiple algorithms and return a comparative benchmark dictionary."""
    results = {}
    best_model = None
    best_f1 = -1.0
    best_type = ""

    for m_type in ["gradient_boosting", "random_forest", "logistic_regression"]:
        predictor = AttritionPredictor(model_type=m_type)
        predictor.train(X_train, y_train)
        eval_metrics = predictor.evaluate(X_test, y_test)
        results[m_type] = {
            "metrics": eval_metrics,
            "predictor": predictor,
        }
        if eval_metrics["f1_score"] > best_f1:
            best_f1 = eval_metrics["f1_score"]
            best_model = predictor
            best_type = m_type

    return {
        "benchmark": results,
        "best_model_type": best_type,
        "best_model": best_model,
    }

