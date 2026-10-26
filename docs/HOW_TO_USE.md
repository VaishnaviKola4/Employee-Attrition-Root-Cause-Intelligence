# Step-by-Step Operations & Execution Guide

This guide walks through configuring, executing, training, and utilizing the Employee Attrition Root-Cause Intelligence system.

---

## 1. Prerequisites and Installation
Ensure Python 3.9+ is installed on your operating system.

```bash
# Clone repository
git clone https://github.com/VaishnaviKola4/employee-attrition-root-cause-intelligence.git
cd employee-attrition-root-cause-intelligence

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 2. Generating or Ingesting Data
The system accepts standard CSV files containing HR metrics or can generate realistic synthetic datasets for demonstration.

### Option A: Automatic Synthetic Generation
To generate 1,500 anonymized records matching multi-source corporate dynamics:
```bash
python -m src.main --generate --samples 1500 --data data/raw/employee_data.csv
```

### Option B: Providing Custom HR Data
Place your internal anonymized CSV dataset at `data/raw/employee_data.csv`. Refer to `README.md` for the required column schema and data types.

---

## 3. Training & Evaluating Models
To execute the end-to-end preprocessing, model benchmark, and statistical root-cause pipeline:
```bash
python -m src.main
```
This performs:
1. Feature engineering (burnout index, compensation stagnation index, career velocity).
2. One-hot encoding & standard scaling.
3. Comparative training across Gradient Boosting, Random Forest, and Logistic Regression.
4. Exporting serialized artifacts to `src/models/saved_models/`.
5. Generating statistical reports at `docs/attrition_intelligence_report.json`.

---

## 4. Running the Interactive Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501` to view:
- **Executive Dashboard:** Macro attrition metrics and replacement cost exposure.
- **Root Cause Diagnostics:** Statistical odds ratios, t-tests, and chi-square significance.
- **Individual Risk Profiler:** Targeted employee lookups and intervention ROI calculators.
- **What-If Policy Simulator:** Interactive sliders simulating policy changes.
- **Cohort & Seasonal Intelligence:** Dedicated tools for peak season and contract management.

---

## 5. Running the Unit & Pipeline Tests
Validate all modules:
```bash
python -m pytest tests/ -v
```
All tests verify data generation schema, feature engineering, model training, statistical tests, and recommendation engines.

