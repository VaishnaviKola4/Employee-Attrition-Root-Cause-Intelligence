# API & Interface Reference Guide

This document details the software interfaces, programmatic APIs, and interactive UI views provided by the system.

---

## 1. Programmatic Python API

### `HRDataGenerator` (`src.data.generator`)
Generates anonymized synthetic workforce datasets with correlated multi-source distributions.

```python
from src.data.generator import HRDataGenerator

generator = HRDataGenerator(random_seed=42)
df = generator.generate_workforce_data(n_samples=1000)
```

### `DataPreprocessor` (`src.data.preprocessor`)
Applies domain-specific feature engineering, scaling, and categorical encoding.

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X, y, feature_names = preprocessor.fit_transform(df, target_col="attrition")

# Transform new unseen batches:
X_unseen = preprocessor.transform(new_df)
```

### `AttritionPredictor` (`src.models.train`)
Unified wrapper supporting `gradient_boosting`, `random_forest`, and `logistic_regression`.

```python
from src.models.train import AttritionPredictor

predictor = AttritionPredictor(model_type="gradient_boosting")
predictor.train(X_train, y_train)

# Inference
probabilities = predictor.predict_proba(X_test)
evaluation_metrics = predictor.evaluate(X_test, y_test)
```

### `RootCauseAnalyzer` (`src.analysis.root_cause`)
Performs statistical significance tests and odds ratio estimations.

```python
from src.analysis.root_cause import RootCauseAnalyzer

analyzer = RootCauseAnalyzer(df, target_col="attrition")

# Compute actionable threshold Odds Ratios
odds_df = analyzer.compute_odds_ratios()

# Continuous feature statistical comparison (T-test & Mann-Whitney U)
num_stats = analyzer.continuous_feature_tests(["overtime_hours_month", "compa_ratio"])

# Categorical Independence Test (Chi-Square)
cat_stats = analyzer.categorical_chi2_tests(["department", "contract_type"])

# Cohort Risk Segmentation
cohorts = analyzer.identify_high_risk_cohorts()
```

### `RetentionRecommender` (`src.recommender.engine`)
Generates actionable retention playbooks and computes financial ROI.

```python
from src.recommender.engine import RetentionRecommender

recommender = RetentionRecommender()
playbook = recommender.generate_individual_recommendations(employee_series, predicted_risk=0.82)

# Strategic summary
org_strategy = recommender.generate_organizational_strategy(df, probabilities)
```

---

## 2. Interactive User Interface Structure (`src/app.py`)

The Streamlit web application is organized into 5 primary operational views:

| View | Purpose | Key Components |
|---|---|---|
| **Executive Dashboard** | C-suite & VP People overview | Headcount KPIs, Flight Risk %, Replacement cost exposure, Department risk charts. |
| **Root Cause Diagnostics** | People Analytics investigation | Horizontal Odds Ratio chart, T-test mean differences table, Chi-square independence. |
| **Individual Risk & Playbooks** | HRBP & Frontline Manager intervention | Employee selector, risk score, diagnostic drivers, step-by-step playbooks, ROI projection. |
| **What-If Policy Simulator** | Executive strategic workforce planning | Real-time policy toggles (overtime cap, salary parity, remote flexibility) recalculating workforce risk and cost savings. |
| **Cohort & Seasonal Intelligence** | Contract & peak operations management | Contract type comparison, tenure milestone curves, seasonal surge intervention framework. |

