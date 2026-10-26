"""
Comprehensive test suite for Data Generator, Preprocessor, Predictive Model,
Root Cause Analyzer, and Recommendation Engine.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.generator import HRDataGenerator, download_or_generate_dataset
from src.data.preprocessor import DataPreprocessor, prepare_train_test_split
from src.models.train import AttritionPredictor, train_and_compare_models
from src.analysis.root_cause import RootCauseAnalyzer
from src.recommender.engine import RetentionRecommender


@pytest.fixture
def sample_dataset():
    generator = HRDataGenerator(random_seed=42)
    return generator.generate_workforce_data(n_samples=150)


def test_data_generator_output(sample_dataset):
    df = sample_dataset
    assert len(df) == 150
    assert "employee_id" in df.columns
    assert "attrition" in df.columns
    assert df["attrition"].isin([0, 1]).all()
    assert df["compa_ratio"].min() >= 0.5
    assert df["monthly_income"].min() > 1000
    # Check anonymization
    assert df["employee_id"].iloc[0].startswith("EMP_")


def test_preprocessor_pipeline(sample_dataset):
    preprocessor = DataPreprocessor()
    X, y, feat_names = preprocessor.fit_transform(sample_dataset)

    assert X.shape[0] == len(sample_dataset)
    assert X.shape[1] == len(feat_names)
    assert not np.isnan(X).any()
    assert len(y) == len(sample_dataset)

    # Test transform on unseen copy
    unseen_df = sample_dataset.copy().iloc[:10]
    X_unseen = preprocessor.transform(unseen_df)
    assert X_unseen.shape[0] == 10
    assert X_unseen.shape[1] == X.shape[1]


def test_model_training_and_evaluation(sample_dataset):
    split = prepare_train_test_split(sample_dataset, test_size=0.3, random_state=42)
    predictor = AttritionPredictor(model_type="logistic_regression")
    predictor.train(split["X_train"], split["y_train"])

    preds = predictor.predict(split["X_test"])
    probs = predictor.predict_proba(split["X_test"])

    assert len(preds) == len(split["y_test"])
    assert len(probs) == len(split["y_test"])
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    eval_res = predictor.evaluate(split["X_test"], split["y_test"])
    assert "accuracy" in eval_res
    assert "roc_auc" in eval_res
    assert eval_res["roc_auc"] >= 0.5


def test_root_cause_analyzer(sample_dataset):
    analyzer = RootCauseAnalyzer(sample_dataset)
    
    odds_df = analyzer.compute_odds_ratios()
    assert not odds_df.empty
    assert "odds_ratio" in odds_df.columns
    assert "risk_factor" in odds_df.columns

    stat_cont = analyzer.continuous_feature_tests(["overtime_hours_month", "compa_ratio"])
    assert len(stat_cont) == 2
    assert "p_value" in stat_cont.columns

    cohorts = analyzer.identify_high_risk_cohorts()
    assert len(cohorts) > 0
    assert "cohort_name" in cohorts[0]


def test_recommendation_engine(sample_dataset):
    recommender = RetentionRecommender()
    high_risk_emp = sample_dataset.iloc[0].copy()
    high_risk_emp["overtime_hours_month"] = 40.0
    high_risk_emp["work_life_balance"] = 1
    high_risk_emp["compa_ratio"] = 0.75

    rec = recommender.generate_individual_recommendations(high_risk_emp, predicted_risk=0.88)
    assert rec["risk_level"] == "CRITICAL"
    assert len(rec["recommended_interventions"]) >= 2
    assert rec["projected_intervention_roi"] > 0
    assert rec["estimated_intervention_cost"] > 0

