"""
Synthetic HR Dataset Generator & Ingestion Module.
Generates multi-source realistic HR enterprise data:
- Performance & Evaluations
- Workload & Overtime
- Compensation & Market Ratio
- Absenteeism & Sick Leave
- Engagement & Pulse Surveys
- Training & Professional Development
- Contract & Demographics
"""

import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class HRDataGenerator:
    """Generates synthetic, anonymized, and statistically realistic enterprise HR data."""

    DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Customer Support", "Product", "Operations", "HR"]
    ROLES = {
        "Engineering": ["Junior Engineer", "Software Engineer", "Senior Engineer", "Tech Lead", "Engineering Manager"],
        "Sales": ["Sales Associate", "Account Executive", "Senior AE", "Sales Manager", "Sales Director"],
        "Marketing": ["Marketing Associate", "Content Strategist", "Growth Marketer", "Marketing Manager"],
        "Customer Support": ["Support Specialist", "Tier 2 Support", "Support Lead", "Support Operations Manager"],
        "Product": ["Associate PM", "Product Manager", "Senior PM", "Group PM"],
        "Operations": ["Operations Analyst", "Operations Coordinator", "Operations Manager"],
        "HR": ["HR Coordinator", "HR Business Partner", "Talent Acquisition Lead", "HR Director"],
    }
    EDUCATION_LEVELS = ["Bachelor's", "Master's", "PhD", "Associate", "High School"]
    CONTRACT_TYPES = ["Full-Time", "Seasonal/Contract", "Part-Time", "Intern"]

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    @staticmethod
    def anonymize_id(employee_id_num: int) -> str:
        """Create a cryptographic anonymized hash for employee identifiers."""
        raw = f"EMP-{employee_id_num:05d}-SALT-9821"
        return f"EMP_{hashlib.sha256(raw.encode()).hexdigest()[:10].upper()}"

    def generate_workforce_data(self, n_samples: int = 1500) -> pd.DataFrame:
        """
        Generate integrated multi-source employee attrition dataset.
        Includes realistic multi-collinear and root-cause drivers of turnover.
        """
        records = []
        for i in range(1, n_samples + 1):
            emp_id = self.anonymize_id(i)
            dept = random.choice(self.DEPARTMENTS)
            role = random.choice(self.ROLES[dept])
            contract = np.random.choice(self.CONTRACT_TYPES, p=[0.75, 0.15, 0.07, 0.03])
            tenure_months = int(np.clip(np.random.exponential(scale=32), 1, 140))
            age = int(np.clip(22 + (tenure_months // 12) + np.random.normal(6, 4), 21, 65))
            education = random.choice(self.EDUCATION_LEVELS)

            # Workload & Overtime
            base_weekly_hours = 40.0
            overtime_prob = 0.35 if dept in ["Engineering", "Sales", "Customer Support"] else 0.20
            has_regular_overtime = np.random.rand() < overtime_prob
            weekly_hours = round(
                base_weekly_hours + (np.random.gamma(shape=3.0, scale=3.5) if has_regular_overtime else np.random.normal(0, 1.5)),
                1
            )
            weekly_hours = max(30.0, weekly_hours)
            overtime_hours_month = max(0.0, round((weekly_hours - 40.0) * 4.33, 1))

            # Compensation metrics
            base_salary_tier = {
                "Engineering": 95000,
                "Product": 92000,
                "Sales": 75000,
                "Marketing": 70000,
                "Operations": 65000,
                "HR": 65000,
                "Customer Support": 52000,
            }[dept]
            salary_multiplier = 1.0 + (tenure_months / 12) * 0.04 + (0.3 if "Manager" in role or "Lead" in role or "Director" in role else 0.0)
            monthly_income = int(np.clip(np.random.normal(base_salary_tier * salary_multiplier / 12, 600), 2500, 22000))
            
            # Compa-ratio: ratio of actual salary to market benchmark (1.0 = at market, <0.85 = underpaid)
            compa_ratio = round(np.clip(np.random.normal(0.98, 0.12), 0.65, 1.45), 2)
            last_raise_months_ago = int(np.clip(np.random.exponential(scale=14), 1, 48))

            # Performance & Promotion
            performance_rating = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.50, 0.22, 0.08]), 1, 5))
            months_since_last_promotion = int(np.clip(np.random.exponential(scale=20), 0, tenure_months))
            projects_completed = int(np.clip(np.random.poisson(lam=4.2 * (tenure_months / 12 + 0.5)), 0, 40))

            # Engagement & Survey Pulse (Scores 1 to 5)
            # Stagnation and low compa-ratio decrease engagement
            job_satisfaction = int(np.clip(
                np.round(3.2 + (compa_ratio - 1.0) * 2.0 - (1 if overtime_hours_month > 25 else 0) + np.random.normal(0, 0.7)),
                1, 5
            ))
            work_life_balance = int(np.clip(
                np.round(4.0 - (overtime_hours_month / 15.0) + np.random.normal(0, 0.6)),
                1, 5
            ))
            manager_relationship = int(np.clip(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.14, 0.38, 0.28, 0.12]), 1, 5))

            # Absenteeism & Sick Leave (days/year)
            sick_leave_days = int(np.clip(
                np.random.poisson(lam=4.0 + (2.5 if work_life_balance <= 2 else 0) + (1.5 if job_satisfaction <= 2 else 0)),
                0, 30
            ))
            unexcused_absences = int(np.clip(np.random.poisson(lam=0.6 if job_satisfaction <= 2 else 0.1), 0, 10))

            # Training & Professional Development
            training_hours_completed = int(np.clip(np.random.normal(24, 12), 0, 80))
            career_growth_opportunities = int(np.clip(
                np.round(2.8 + (1.0 if months_since_last_promotion < 18 else -0.8) + (0.5 if training_hours_completed > 30 else -0.3) + np.random.normal(0, 0.6)),
                1, 5
            ))

            # Commute & Remote setup
            commute_distance_km = int(np.clip(np.random.exponential(scale=16), 1, 75))
            remote_work_pct = np.random.choice([0, 20, 50, 80, 100], p=[0.25, 0.15, 0.30, 0.20, 0.10])

            # Seasonal peak pressure (relevant for seasonal contracts or retail/Q4 spikes)
            is_peak_season_exposure = 1 if (contract == "Seasonal/Contract" or dept in ["Sales", "Customer Support"]) and np.random.rand() < 0.65 else 0

            # Ground truth root cause synthesis for attrition probability
            # Multi-factorial logit score
            logit = -2.8  # baseline attrition around 15-18%
            logit += 1.8 * (1 if overtime_hours_month > 28 else 0)
            logit += 1.6 * (1 if compa_ratio < 0.85 else 0)
            logit += 1.3 * (1 if career_growth_opportunities <= 2 and tenure_months >= 24 else 0)
            logit += 1.2 * (1 if work_life_balance <= 2 else 0)
            logit += 1.1 * (1 if manager_relationship <= 2 else 0)
            logit += 0.9 * (1 if last_raise_months_ago > 24 else 0)
            logit += 1.2 * (1 if contract == "Seasonal/Contract" and is_peak_season_exposure else 0)
            logit += 0.7 * (1 if commute_distance_km > 35 and remote_work_pct == 0 else 0)
            logit -= 1.1 * (1 if training_hours_completed >= 35 else 0)
            logit -= 0.8 * (1 if compa_ratio >= 1.1 else 0)

            # High performers underpaid have high attrition risk (poaching)
            if performance_rating >= 4 and compa_ratio < 0.90:
                logit += 1.4

            prob = 1.0 / (1.0 + np.exp(-logit))
            attrition = 1 if np.random.rand() < prob else 0

            records.append({
                "employee_id": emp_id,
                "department": dept,
                "job_role": role,
                "contract_type": contract,
                "age": age,
                "tenure_months": tenure_months,
                "education_level": education,
                "monthly_income": monthly_income,
                "compa_ratio": compa_ratio,
                "last_raise_months_ago": last_raise_months_ago,
                "weekly_hours": weekly_hours,
                "overtime_hours_month": overtime_hours_month,
                "performance_rating": performance_rating,
                "projects_completed": projects_completed,
                "months_since_last_promotion": months_since_last_promotion,
                "job_satisfaction": job_satisfaction,
                "work_life_balance": work_life_balance,
                "manager_relationship": manager_relationship,
                "career_growth_opportunities": career_growth_opportunities,
                "sick_leave_days": sick_leave_days,
                "unexcused_absences": unexcused_absences,
                "training_hours_completed": training_hours_completed,
                "commute_distance_km": commute_distance_km,
                "remote_work_pct": remote_work_pct,
                "peak_season_pressure": is_peak_season_exposure,
                "attrition": attrition,
            })

        df = pd.DataFrame(records)
        return df


def download_or_generate_dataset(output_path: str = "data/raw/employee_data.csv", n_samples: int = 1500) -> pd.DataFrame:
    """Generate or download synthetic enterprise attrition data and save locally."""
    generator = HRDataGenerator(random_seed=42)
    df = generator.generate_workforce_data(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = download_or_generate_dataset()
    print(f"Generated {len(df)} employee records. Attrition rate: {df['attrition'].mean():.2%}")

