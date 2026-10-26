"""
Data Preprocessing, Cleaning, Anonymization, and Feature Engineering Pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    """Preprocesses and transforms raw workforce datasets for machine learning."""

    CATEGORICAL_COLS = ["department", "job_role", "contract_type", "education_level"]
    NUMERICAL_COLS = [
        "age",
        "tenure_months",
        "monthly_income",
        "compa_ratio",
        "last_raise_months_ago",
        "weekly_hours",
        "overtime_hours_month",
        "performance_rating",
        "projects_completed",
        "months_since_last_promotion",
        "job_satisfaction",
        "work_life_balance",
        "manager_relationship",
        "career_growth_opportunities",
        "sick_leave_days",
        "unexcused_absences",
        "training_hours_completed",
        "commute_distance_km",
        "remote_work_pct",
        "peak_season_pressure",
    ]

    def __init__(self):
        self.column_transformer: ColumnTransformer = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive meaningful HR ratio and risk interaction features."""
        df_feat = df.copy()

        # Interaction: Burnout index = (weekly_hours / 40.0) * (6 - work_life_balance)
        df_feat["burnout_risk_score"] = (df_feat["weekly_hours"] / 40.0) * (6 - df_feat["work_life_balance"])

        # Interaction: Compensation dissatisfaction = (1.0 - compa_ratio) * (df_feat["last_raise_months_ago"] / 12.0)
        df_feat["comp_stagnation_index"] = np.maximum(0, 1.1 - df_feat["compa_ratio"]) * (df_feat["last_raise_months_ago"] / 12.0)

        # Career velocity = projects_completed / (tenure_months / 12.0 + 0.5)
        df_feat["career_velocity"] = df_feat["projects_completed"] / (df_feat["tenure_months"] / 12.0 + 0.5)

        # Disengagement indicator: unexcused absences + (6 - job_satisfaction)
        df_feat["disengagement_score"] = df_feat["unexcused_absences"] * 1.5 + (5 - df_feat["job_satisfaction"])

        return df_feat

    def get_all_feature_cols(self) -> Tuple[List[str], List[str]]:
        engineered_numeric = [
            "burnout_risk_score",
            "comp_stagnation_index",
            "career_velocity",
            "disengagement_score",
        ]
        all_numeric = self.NUMERICAL_COLS + engineered_numeric
        return self.CATEGORICAL_COLS, all_numeric

    def fit_transform(self, df: pd.DataFrame, target_col: str = "attrition") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit preprocessor on dataset and return transformed features, labels, and feature names."""
        df_processed = self.engineer_features(df)
        cat_cols, num_cols = self.get_all_feature_cols()

        X_raw = df_processed[cat_cols + num_cols]
        y = df_processed[target_col].values if target_col in df_processed.columns else None

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        X_trans = self.column_transformer.fit_transform(X_raw)

        # Retrieve feature names
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        self.feature_names = num_cols + encoded_cat_names
        self.is_fitted = True

        return X_trans, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform unseen data with fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
        df_processed = self.engineer_features(df)
        cat_cols, num_cols = self.get_all_feature_cols()
        X_raw = df_processed[cat_cols + num_cols]
        return self.column_transformer.transform(X_raw)


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """Preprocess data and return train/test splits along with preprocessor."""
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }

