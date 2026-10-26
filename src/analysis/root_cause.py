"""
Root-Cause Statistical Analysis Module.
Implements:
1. Chi-Square Test of Independence for categorical variables.
2. Two-sample Mann-Whitney U & Student's T-Tests for continuous metrics.
3. Pearson & Spearman correlation analysis with attrition.
4. Odds Ratios and Risk Ratio analysis.
5. Multivariate Attribution & High-Risk Cohort Segmentation.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats


class RootCauseAnalyzer:
    """Discovers statistical drivers and root causes of employee turnover."""

    def __init__(self, df: pd.DataFrame, target_col: str = "attrition"):
        self.df = df.copy()
        self.target_col = target_col
        self.attrited = self.df[self.df[target_col] == 1]
        self.retained = self.df[self.df[target_col] == 0]

    def continuous_feature_tests(self, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Perform t-test and Mann-Whitney U test on numerical metrics between
        attrited and retained employee groups.
        """
        results = []
        for col in numeric_cols:
            if col not in self.df.columns or col == self.target_col:
                continue

            vals_att = self.attrited[col].dropna()
            vals_ret = self.retained[col].dropna()

            mean_att = float(vals_att.mean())
            mean_ret = float(vals_ret.mean())
            diff = mean_att - mean_ret
            pct_diff = (diff / mean_ret * 100) if mean_ret != 0 else 0

            # Independent T-Test
            t_stat, t_pval = stats.ttest_ind(vals_att, vals_ret, equal_var=False)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(vals_att, vals_ret, alternative="two-sided")

            # Point-biserial correlation
            corr, corr_pval = stats.pointbiserialr(self.df[col], self.df[self.target_col])

            results.append({
                "feature": col,
                "mean_attrited": round(mean_att, 2),
                "mean_retained": round(mean_ret, 2),
                "absolute_diff": round(diff, 2),
                "pct_difference": round(pct_diff, 1),
                "t_statistic": round(float(t_stat), 3),
                "p_value": float(t_pval),
                "mann_whitney_p": float(u_pval),
                "correlation": round(float(corr), 3),
                "is_significant_p05": bool(t_pval < 0.05),
            })

        df_res = pd.DataFrame(results)
        return df_res.sort_values(by="p_value", ascending=True)

    def categorical_chi2_tests(self, cat_cols: List[str]) -> pd.DataFrame:
        """
        Perform Chi-square test of independence and compute attrition rate by group.
        """
        results = []
        for col in cat_cols:
            if col not in self.df.columns:
                continue

            contingency = pd.crosstab(self.df[col], self.df[self.target_col])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

            # Group attrition rates
            group_rates = self.df.groupby(col)[self.target_col].agg(["count", "mean"])
            highest_group = group_rates["mean"].idxmax()
            highest_rate = float(group_rates["mean"].max())
            lowest_group = group_rates["mean"].idxmin()
            lowest_rate = float(group_rates["mean"].min())

            results.append({
                "categorical_variable": col,
                "chi2_statistic": round(float(chi2), 3),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_val),
                "highest_risk_segment": f"{highest_group} ({highest_rate:.1%})",
                "lowest_risk_segment": f"{lowest_group} ({lowest_rate:.1%})",
                "is_significant_p05": bool(p_val < 0.05),
            })

        df_res = pd.DataFrame(results)
        return df_res.sort_values(by="chi2_statistic", ascending=False)

    def compute_odds_ratios(self) -> pd.DataFrame:
        """
        Calculate binary Odds Ratios (OR) for actionable risk thresholds:
        - Excessive Overtime (> 25 hrs/mo)
        - Sub-Market Compensation (compa_ratio < 0.85)
        - Stagnant Career (> 24 months without raise or promotion)
        - Low Satisfaction (job_satisfaction <= 2)
        - Poor Work-Life Balance (work_life_balance <= 2)
        - Seasonal/Contract Employment
        """
        conditions = {
            "High Overtime (> 25h/mo)": self.df["overtime_hours_month"] > 25,
            "Underpaid (Compa Ratio < 0.85)": self.df["compa_ratio"] < 0.85,
            "Low Job Satisfaction (<= 2)": self.df["job_satisfaction"] <= 2,
            "Poor Work-Life Balance (<= 2)": self.df["work_life_balance"] <= 2,
            "Stagnant Compensation (> 24m no raise)": self.df["last_raise_months_ago"] > 24,
            "Poor Manager Relationship (<= 2)": self.df["manager_relationship"] <= 2,
            "Low Training (< 15h/yr)": self.df["training_hours_completed"] < 15,
            "Seasonal/Contract Type": self.df["contract_type"] == "Seasonal/Contract",
            "Long Commute (> 35km without remote)": (self.df["commute_distance_km"] > 35) & (self.df["remote_work_pct"] == 0),
        }

        results = []
        for name, cond in conditions.items():
            ct = pd.crosstab(cond, self.df[self.target_col])
            # ct index: False, True; columns: 0 (retained), 1 (attrited)
            try:
                a = ct.loc[True, 1] if True in ct.index and 1 in ct.columns else 0
                b = ct.loc[True, 0] if True in ct.index and 0 in ct.columns else 0
                c = ct.loc[False, 1] if False in ct.index and 1 in ct.columns else 0
                d = ct.loc[False, 0] if False in ct.index and 0 in ct.columns else 0

                # Odds Ratio = (a * d) / (b * c)
                if (b * c) > 0:
                    odds_ratio = (a * d) / (b * c)
                    log_or = np.log(odds_ratio)
                    se_log_or = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
                    ci_lower = np.exp(log_or - 1.96 * se_log_or)
                    ci_upper = np.exp(log_or + 1.96 * se_log_or)
                else:
                    odds_ratio, ci_lower, ci_upper = np.nan, np.nan, np.nan

                # Risk in exposed vs unexposed
                risk_exposed = a / (a + b) if (a + b) > 0 else 0
                risk_unexposed = c / (c + d) if (c + d) > 0 else 0
                relative_risk = risk_exposed / risk_unexposed if risk_unexposed > 0 else np.nan

                results.append({
                    "risk_factor": name,
                    "odds_ratio": round(float(odds_ratio), 2) if not np.isnan(odds_ratio) else 1.0,
                    "ci_95_range": f"[{ci_lower:.2f} - {ci_upper:.2f}]" if not np.isnan(ci_lower) else "N/A",
                    "attrition_rate_exposed": f"{risk_exposed:.1%}",
                    "attrition_rate_unexposed": f"{risk_unexposed:.1%}",
                    "relative_risk": round(float(relative_risk), 2) if not np.isnan(relative_risk) else 1.0,
                    "affected_headcount": int(a + b),
                })
            except Exception:
                continue

        df_or = pd.DataFrame(results)
        return df_or.sort_values(by="odds_ratio", ascending=False)

    def identify_high_risk_cohorts(self) -> List[Dict[str, Any]]:
        """Identify multi-factorial vulnerable segments within the workforce."""
        cohorts = [
            {
                "cohort_name": "Burned-Out Peak Contributors",
                "condition": (self.df["overtime_hours_month"] > 25) & (self.df["work_life_balance"] <= 2),
                "primary_driver": "Chronic workload exhaustion, sustained high overtime hours",
                "urgency": "CRITICAL",
            },
            {
                "cohort_name": "Under-Market High Performers (Flight Risk)",
                "condition": (self.df["performance_rating"] >= 4) & (self.df["compa_ratio"] < 0.88),
                "primary_driver": "Salary below market benchmark despite high merit/performance",
                "urgency": "CRITICAL",
            },
            {
                "cohort_name": "Stagnant Mid-Tenure Employees",
                "condition": (self.df["tenure_months"] >= 24) & (self.df["months_since_last_promotion"] >= 24) & (self.df["career_growth_opportunities"] <= 2),
                "primary_driver": "Lack of upward mobility, promotion blockage, limited career growth",
                "urgency": "HIGH",
            },
            {
                "cohort_name": "Seasonal & Peak-Season Attrition",
                "condition": (self.df["contract_type"] == "Seasonal/Contract") & (self.df["peak_season_pressure"] == 1),
                "primary_driver": "Peak season workload spikes without onboarding support or contract renewal clarity",
                "urgency": "HIGH",
            },
            {
                "cohort_name": "Disengaged Onboarding / Early Career",
                "condition": (self.df["tenure_months"] <= 12) & ((self.df["job_satisfaction"] <= 2) | (self.df["manager_relationship"] <= 2)),
                "primary_driver": "Onboarding disconnect, poor manager alignment, culture misfit",
                "urgency": "MEDIUM",
            },
        ]

        summary = []
        for c in cohorts:
            subset = self.df[c["condition"]]
            total_in_cohort = len(subset)
            if total_in_cohort > 0:
                attrition_count = subset[self.target_col].sum()
                rate = attrition_count / total_in_cohort
                summary.append({
                    "cohort_name": c["cohort_name"],
                    "urgency": c["urgency"],
                    "total_employees": int(total_in_cohort),
                    "attrition_rate": float(rate),
                    "primary_driver": c["primary_driver"],
                })

        return summary

