"""
Main CLI Pipeline for Employee Attrition Root-Cause Intelligence.
Performs end-to-end data generation, preprocessing, model training,
root cause statistical analysis, and retention playbook generation.
"""

import argparse
import os
import joblib
import pandas as pd

from src.data.generator import download_or_generate_dataset
from src.data.preprocessor import DataPreprocessor, prepare_train_test_split
from src.models.train import AttritionPredictor, train_and_compare_models
from src.analysis.root_cause import RootCauseAnalyzer
from src.recommender.engine import RetentionRecommender
from src.utils.logger import get_logger, save_json

logger = get_logger("main_pipeline")


def run_pipeline(
    data_path: str = "data/raw/employee_data.csv",
    generate_if_missing: bool = True,
    n_samples: int = 1500,
    model_output_path: str = "src/models/saved_models/attrition_model.joblib",
    preprocessor_output_path: str = "src/models/saved_models/preprocessor.joblib",
):
    """Execute end-to-end analytics and model training pipeline."""
    logger.info("Initializing Employee Attrition Root-Cause Intelligence Pipeline...")

    # Step 1: Data Collection & Preparation
    if not os.path.exists(data_path) and generate_if_missing:
        logger.info(f"Dataset not found at {data_path}. Generating synthetic enterprise workforce data ({n_samples} records)...")
        df = download_or_generate_dataset(data_path, n_samples=n_samples)
    else:
        logger.info(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Baseline attrition rate: {df['attrition'].mean():.2%}")

    # Step 2: Data Preprocessing & Splitting
    logger.info("Engineering domain features and encoding datasets...")
    split_data = prepare_train_test_split(df, test_size=0.2, random_state=42)
    X_train, X_test = split_data["X_train"], split_data["X_test"]
    y_train, y_test = split_data["y_train"], split_data["y_test"]
    feature_names = split_data["feature_names"]
    preprocessor = split_data["preprocessor"]

    # Step 3: Model Training and Comparative Benchmarking
    logger.info("Benchmarking machine learning classifiers (Gradient Boosting, Random Forest, Logistic Regression)...")
    comparison = train_and_compare_models(X_train, y_train, X_test, y_test)
    best_model_type = comparison["best_model_type"]
    best_model: AttritionPredictor = comparison["best_model"]
    logger.info(f"Best performing model: {best_model_type.upper()}")

    for m_type, res in comparison["benchmark"].items():
        m = res["metrics"]
        logger.info(f"Model: {m_type:20s} | ROC-AUC: {m['roc_auc']:.4f} | F1: {m['f1_score']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f}")

    # Save best model & preprocessor
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    best_model.save(model_output_path)
    joblib.dump(preprocessor, preprocessor_output_path)
    logger.info(f"Persisted trained model and preprocessor to {model_output_path} and {preprocessor_output_path}")

    # Step 4: Root-Cause Statistical Analysis
    logger.info("Conducting Root-Cause Statistical Analysis (Chi-square, Mann-Whitney U, Odds Ratios)...")
    rc_analyzer = RootCauseAnalyzer(df)
    cat_cols = ["department", "job_role", "contract_type", "education_level"]
    num_cols = [
        "overtime_hours_month", "compa_ratio", "job_satisfaction", "work_life_balance",
        "monthly_income", "tenure_months", "months_since_last_promotion", "sick_leave_days"
    ]
    stat_continuous = rc_analyzer.continuous_feature_tests(num_cols)
    stat_categorical = rc_analyzer.categorical_chi2_tests(cat_cols)
    odds_ratios = rc_analyzer.compute_odds_ratios()
    cohorts = rc_analyzer.identify_high_risk_cohorts()

    logger.info("--- Top Root Causes by Odds Ratio (Relative Risk of Turnover) ---")
    for _, row in odds_ratios.head(5).iterrows():
        logger.info(f"Risk Factor: {row['risk_factor']:35s} | Odds Ratio: {row['odds_ratio']:.2f} | Exposed Attrition: {row['attrition_rate_exposed']}")

    # Step 5: Recommendation Engine Execution
    logger.info("Generating Retention Intervention Playbooks and ROI projections...")
    recommender = RetentionRecommender()
    all_probs = best_model.predict_proba(preprocessor.transform(df))
    org_strategy = recommender.generate_organizational_strategy(df, all_probs)
    
    logger.info(f"Analysis complete! Critical risk headcount: {org_strategy['critical_risk_employees']}, High risk: {org_strategy['high_risk_employees']}")

    # Save summary report
    report = {
        "dataset_summary": {
            "total_employees": len(df),
            "attrition_rate": float(df["attrition"].mean()),
        },
        "model_performance": comparison["benchmark"][best_model_type]["metrics"],
        "top_odds_ratios": odds_ratios.to_dict(orient="records"),
        "high_risk_cohorts": cohorts,
        "organizational_strategy": org_strategy,
    }
    save_json(report, "docs/attrition_intelligence_report.json")
    logger.info("Report exported to docs/attrition_intelligence_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Attrition Intelligence CLI")
    parser.add_argument("--data", type=str, default="data/raw/employee_data.csv", help="Path to input data")
    parser.add_argument("--generate", action="store_true", default=True, help="Generate synthetic data if missing")
    parser.add_argument("--samples", type=int, default=1500, help="Number of records to generate")
    args = parser.parse_args()

    run_pipeline(data_path=args.data, generate_if_missing=args.generate, n_samples=args.samples)

