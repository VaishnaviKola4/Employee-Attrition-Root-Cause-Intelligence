"""
Evidence-Based HR Retention & Intervention Recommendation Engine.
Analyzes employee risk profiles and root cause drivers to produce targeted,
actionable mitigation strategies with ROI estimates and operational timelines.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


class RetentionRecommender:
    """Produces multi-tier evidence-based intervention playbooks for retention."""

    INTERVENTION_CATALOG = {
        "burnout_mitigation": {
            "title": "Workload Rebalancing & Overtime Cap Protocol",
            "category": "Workload & Wellbeing",
            "impact": "High (Reduces workload-driven attrition by up to 35%)",
            "actions": [
                "Institute mandatory weekly overtime capping (<= 10 hours overtime/month).",
                "Trigger automated workload distribution rebalancing across peer teams.",
                "Provide 'recharge days' and mandatory flexi-hours for high-overtime projects.",
                "Review contractor or temporary staffing support for peak sprint deadlines.",
            ],
            "cost_per_employee": 450,
            "estimated_implementation_days": 14,
        },
        "market_compensation_adjustment": {
            "title": "Proactive Compa-Ratio Parity Realignment",
            "category": "Compensation & Rewards",
            "impact": "Very High (Retains 60-70% of high performers vulnerable to market poaching)",
            "actions": [
                "Adjust compa-ratio to >= 0.95 against regional industry benchmarks.",
                "Fast-track compensation review without waiting for annual cycles.",
                "Grant spot retention equity / milestone-based retention bonus.",
                "Transparently articulate the total rewards structure and promotion timeline.",
            ],
            "cost_per_employee": 4200,
            "estimated_implementation_days": 21,
        },
        "career_growth_mobility": {
            "title": "Targeted Career Pathway & Upskilling Track",
            "category": "Career Progression & Training",
            "impact": "High (Reduces stagnation turnover by ~40%)",
            "actions": [
                "Enroll in individualized leadership development or technical upskilling paths.",
                "Assign an executive mentor from outside their immediate reporting chain.",
                "Establish clear 6-month promotion milestones with defined evaluation criteria.",
                "Offer internal mobility / cross-functional project rotations.",
            ],
            "cost_per_employee": 850,
            "estimated_implementation_days": 30,
        },
        "manager_coaching_alignment": {
            "title": "Manager-Employee Alignment & 1-on-1 Pulse Framework",
            "category": "Leadership & Culture",
            "impact": "Medium-High (Resolves 45% of managerial friction attrition)",
            "actions": [
                "Initiate bi-weekly skip-level syncs with department director.",
                "Provide frontline manager with empathetic leadership coaching.",
                "Conduct structured stay-interviews focused on psychological safety.",
                "Explore lateral department transfer if interpersonal friction persists.",
            ],
            "cost_per_employee": 300,
            "estimated_implementation_days": 7,
        },
        "seasonal_contract_conversion": {
            "title": "Seasonal Peak Transition & Contract Security Incentive",
            "category": "Contract & Seasonal Retention",
            "impact": "Very High (Stabilizes peak season turnover spikes by 50%)",
            "actions": [
                "Offer guaranteed contract-completion bonuses and post-peak retention rewards.",
                "Provide transparent pathway to full-time conversion based on peak performance.",
                "Deploy buddy onboarding system with dedicated tenured shift leads.",
                "Offer shift flexibility and predictable scheduling 2 weeks in advance.",
            ],
            "cost_per_employee": 600,
            "estimated_implementation_days": 10,
        },
        "onboarding_reinforcement": {
            "title": "Structured 90-Day Early Tenure Onboarding Reinforcement",
            "category": "Onboarding & Integration",
            "impact": "High (Cuts first-year employee turnover by 30%)",
            "actions": [
                "Schedule Day 30, 60, and 90 check-ins with People Operations.",
                "Assign an onboard peer buddy to accelerate cultural and tooling navigation.",
                "Clarify expectations and early wins with a tailored 30-60-90 day roadmap.",
                "Regular feedback loops to proactively surface friction or confusion.",
            ],
            "cost_per_employee": 250,
            "estimated_implementation_days": 5,
        },
        "commute_flexibility": {
            "title": "Hybrid Remote-Work Transition & Commute Subsidy",
            "category": "Flexibility & Logistics",
            "impact": "Moderate (Mitigates commute burnout for distant commuters)",
            "actions": [
                "Increase remote work allowance to 3-4 days/week for long-commute roles.",
                "Provide transportation stipend or transit subsidy.",
                "Allow core hours scheduling (10 AM - 4 PM) to avoid rush-hour traffic.",
            ],
            "cost_per_employee": 150,
            "estimated_implementation_days": 7,
        },
    }

    def generate_individual_recommendations(self, employee_row: pd.Series, predicted_risk: float = None) -> Dict[str, Any]:
        """Diagnose root drivers for a single employee and generate tailored mitigations."""
        emp_id = employee_row.get("employee_id", "EMP_UNKNOWN")
        risk = predicted_risk if predicted_risk is not None else employee_row.get("attrition_probability", 0.5)

        interventions = []
        identified_drivers = []

        # Check Overtime & Work-life balance
        if employee_row.get("overtime_hours_month", 0) > 20 or employee_row.get("work_life_balance", 3) <= 2:
            interventions.append(self.INTERVENTION_CATALOG["burnout_mitigation"])
            identified_drivers.append(f"Excessive overtime ({employee_row.get('overtime_hours_month', 0)} hrs/mo) & poor work-life balance ({employee_row.get('work_life_balance', 0)}/5)")

        # Check Compensation & Compa-Ratio
        if employee_row.get("compa_ratio", 1.0) < 0.88 or (employee_row.get("performance_rating", 3) >= 4 and employee_row.get("compa_ratio", 1.0) < 0.95):
            interventions.append(self.INTERVENTION_CATALOG["market_compensation_adjustment"])
            identified_drivers.append(f"Sub-market compensation (Compa-Ratio: {employee_row.get('compa_ratio', 1.0):.2f}) relative to performance rating ({employee_row.get('performance_rating', 0)}/5)")

        # Check Career Stagnation
        if employee_row.get("months_since_last_promotion", 0) >= 24 or employee_row.get("career_growth_opportunities", 3) <= 2:
            interventions.append(self.INTERVENTION_CATALOG["career_growth_mobility"])
            identified_drivers.append(f"Career stagnation ({employee_row.get('months_since_last_promotion', 0)} months without promotion) and low perceived growth")

        # Check Manager Relationship
        if employee_row.get("manager_relationship", 3) <= 2:
            interventions.append(self.INTERVENTION_CATALOG["manager_coaching_alignment"])
            identified_drivers.append(f"Low manager relationship rating ({employee_row.get('manager_relationship', 0)}/5)")

        # Check Seasonal/Contract pressure
        if employee_row.get("contract_type") == "Seasonal/Contract" or employee_row.get("peak_season_pressure", 0) == 1:
            interventions.append(self.INTERVENTION_CATALOG["seasonal_contract_conversion"])
            identified_drivers.append("Seasonal contract volatility / high peak-season stress")

        # Check Early Tenure / Onboarding
        if employee_row.get("tenure_months", 36) <= 12 and employee_row.get("job_satisfaction", 3) <= 2:
            interventions.append(self.INTERVENTION_CATALOG["onboarding_reinforcement"])
            identified_drivers.append("Early-tenure onboarding friction (< 12 months tenure with low satisfaction)")

        # Check Commute
        if employee_row.get("commute_distance_km", 0) > 35 and employee_row.get("remote_work_pct", 0) < 40:
            interventions.append(self.INTERVENTION_CATALOG["commute_flexibility"])
            identified_drivers.append(f"Long daily commute ({employee_row.get('commute_distance_km', 0)} km) without sufficient remote flexibility")

        # Default fallback if no specific rule hit but risk is elevated
        if not interventions and risk > 0.4:
            interventions.append(self.INTERVENTION_CATALOG["manager_coaching_alignment"])
            interventions.append(self.INTERVENTION_CATALOG["career_growth_mobility"])
            identified_drivers.append("Elevated general disengagement signals from multi-factor interactions")

        total_est_cost = sum(i["cost_per_employee"] for i in interventions)
        est_replacement_cost = employee_row.get("monthly_income", 6000) * 6  # standard HR benchmark: 50% annual salary
        expected_roi = ((est_replacement_cost * risk * 0.5) - total_est_cost) / max(1, total_est_cost)

        risk_level = "CRITICAL" if risk >= 0.70 else "HIGH" if risk >= 0.45 else "MODERATE" if risk >= 0.25 else "LOW"

        return {
            "employee_id": emp_id,
            "risk_score": float(risk),
            "risk_level": risk_level,
            "primary_drivers": identified_drivers,
            "recommended_interventions": interventions,
            "estimated_intervention_cost": total_est_cost,
            "estimated_replacement_cost_avoidance": est_replacement_cost,
            "projected_intervention_roi": max(0.0, round(float(expected_roi), 2)),
        }

    def generate_organizational_strategy(self, df: pd.DataFrame, risk_probabilities: np.ndarray) -> Dict[str, Any]:
        """Aggregate department-level insights and company-wide strategic retention priorities."""
        df_copy = df.copy()
        df_copy["attrition_risk"] = risk_probabilities

        dept_summary = (
            df_copy.groupby("department")
            .agg(
                headcount=("employee_id", "count"),
                avg_risk=("attrition_risk", "mean"),
                high_risk_count=("attrition_risk", lambda x: (x >= 0.5).sum()),
                avg_overtime=("overtime_hours_month", "mean"),
                avg_compa=("compa_ratio", "mean"),
                avg_satisfaction=("job_satisfaction", "mean"),
            )
            .reset_index()
        )
        dept_summary["high_risk_pct"] = dept_summary["high_risk_count"] / dept_summary["headcount"]
        dept_summary = dept_summary.sort_values(by="high_risk_pct", ascending=False)

        top_company_drivers = []
        if (df_copy["overtime_hours_month"] > 25).mean() > 0.2:
            top_company_drivers.append("Overwork / Overtime Spike across key delivery units")
        if (df_copy["compa_ratio"] < 0.88).mean() > 0.15:
            top_company_drivers.append("Compensation Gap below market benchmark")
        if (df_copy["months_since_last_promotion"] > 24).mean() > 0.25:
            top_company_drivers.append("Career Mobility & Promotion Bottlenecks")

        return {
            "department_breakdown": dept_summary.to_dict(orient="records"),
            "systemic_drivers": top_company_drivers,
            "total_evaluated": len(df_copy),
            "critical_risk_employees": int((risk_probabilities >= 0.70).sum()),
            "high_risk_employees": int(((risk_probabilities >= 0.45) & (risk_probabilities < 0.70)).sum()),
        }

