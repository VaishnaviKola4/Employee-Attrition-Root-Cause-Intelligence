"""
Interactive Enterprise Streamlit Dashboard for Employee Attrition Root-Cause Intelligence.
Features:
- Executive KPI Overview & Flight Risk Summary
- Department & Cohort Drilldown
- Root-Cause Statistical Analysis (Chi-Square, Odds Ratios, Feature Importances)
- Individual Employee Risk Profiler & Retention Intervention Simulator
- What-If Intervention Policy Simulator
- Exportable HR Retention Action Plans
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.preprocessor import DataPreprocessor
from src.analysis.root_cause import RootCauseAnalyzer
from src.recommender.engine import RetentionRecommender
from src.models.train import AttritionPredictor

# Page Setup
st.set_page_config(
    page_title="Workforce Intelligence | Attrition & Root-Cause AI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
    }
    .badge-critical { background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem; }
    .badge-high { background-color: #ffedd5; color: #9a3412; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem; }
    .badge-moderate { background-color: #fef9c3; color: #854d0e; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem; }
    .badge-low { background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data_path = "data/raw/employee_data.csv"
    if not os.path.exists(data_path):
        from src.data.generator import download_or_generate_dataset
        df = download_or_generate_dataset(data_path, n_samples=1500)
    else:
        df = pd.read_csv(data_path)
    return df


@st.cache_resource
def load_model_and_preprocessor():
    model_path = "src/models/saved_models/attrition_model.joblib"
    prep_path = "src/models/saved_models/preprocessor.joblib"
    if os.path.exists(model_path) and os.path.exists(prep_path):
        predictor = AttritionPredictor(model_type="logistic_regression")
        predictor.load(model_path)
        preprocessor = joblib.load(prep_path)
        return predictor, preprocessor
    return None, None


df = load_data()
predictor, preprocessor = load_model_and_preprocessor()

if predictor is not None and preprocessor is not None:
    X_trans = preprocessor.transform(df)
    df["predicted_risk"] = predictor.predict_proba(X_trans)
else:
    df["predicted_risk"] = df["attrition"].apply(lambda x: 0.85 if x == 1 else 0.15)

recommender = RetentionRecommender()
analyzer = RootCauseAnalyzer(df)

# Sidebar Controls
st.sidebar.title("👥 Workforce AI Navigation")
nav = st.sidebar.radio(
    "Select View",
    [
        "Executive Dashboard",
        "Root Cause Diagnostics",
        "Individual Risk & Playbooks",
        "What-If Policy Simulator",
        "Cohort & Seasonal Intelligence",
    ],
)

st.sidebar.divider()
selected_dept = st.sidebar.multiselect(
    "Filter by Department",
    options=list(df["department"].unique()),
    default=list(df["department"].unique()),
)
filtered_df = df[df["department"].isin(selected_dept)].copy()


# ---------------- PAGE 1: EXECUTIVE DASHBOARD ----------------
if nav == "Executive Dashboard":
    st.title("📊 Executive Attrition & Retention Overview")
    st.markdown("High-level visibility into workforce attrition signals, turnover costs, and root drivers.")

    col1, col2, col3, col4, col5 = st.columns(5)
    total_hc = len(filtered_df)
    critical_hc = (filtered_df["predicted_risk"] >= 0.70).sum()
    high_hc = ((filtered_df["predicted_risk"] >= 0.45) & (filtered_df["predicted_risk"] < 0.70)).sum()
    avg_risk = filtered_df["predicted_risk"].mean()
    potential_turnover_cost = int((critical_hc + high_hc) * filtered_df["monthly_income"].mean() * 6)

    col1.metric("Total Headcount", f"{total_hc:,}")
    col2.metric("Critical Flight Risk", f"{critical_hc:,}", f"{critical_hc/total_hc:.1%}", delta_color="inverse")
    col3.metric("High Flight Risk", f"{high_hc:,}", f"{high_hc/total_hc:.1%}", delta_color="inverse")
    col4.metric("Mean Attrition Risk", f"{avg_risk:.1%}")
    col5.metric("At-Risk Replacement Cost", f"${potential_turnover_cost:,.0f}")

    st.markdown("---")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Departmental Flight Risk & Exposure")
        dept_summary = (
            filtered_df.groupby("department")
            .agg(
                headcount=("employee_id", "count"),
                avg_risk=("predicted_risk", "mean"),
                critical_count=("predicted_risk", lambda x: (x >= 0.7).sum()),
            )
            .reset_index()
        )
        fig_dept = px.bar(
            dept_summary,
            x="department",
            y="avg_risk",
            color="avg_risk",
            color_continuous_scale="Reds",
            title="Average Predicted Attrition Risk by Department",
            labels={"avg_risk": "Mean Risk Score", "department": "Department"},
        )
        fig_dept.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_dept, use_container_width=True)

    with c2:
        st.subheader("Risk Distribution by Job Role")
        role_summary = filtered_df.groupby(["job_role", "department"])["predicted_risk"].mean().reset_index()
        fig_role = px.scatter(
            role_summary,
            x="job_role",
            y="predicted_risk",
            size="predicted_risk",
            color="department",
            title="Attrition Risk vs Job Roles",
        )
        fig_role.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
        st.plotly_chart(fig_role, use_container_width=True)

    st.subheader("Top Systemic Risk Drivers Across Organization")
    odds_table = analyzer.compute_odds_ratios().head(6)
    st.dataframe(
        odds_table[["risk_factor", "odds_ratio", "ci_95_range", "attrition_rate_exposed", "relative_risk", "affected_headcount"]],
        use_container_width=True,
    )


# ---------------- PAGE 2: ROOT CAUSE DIAGNOSTICS ----------------
elif nav == "Root Cause Diagnostics":
    st.title("🔬 Root-Cause Statistical Analysis")
    st.markdown("Rigorous statistical tests identifying the true statistical causes of turnover.")

    tab1, tab2, tab3 = st.tabs(["Odds Ratios & Relative Risk", "Continuous Drivers (T-Test / Mann-Whitney)", "Categorical Independence (Chi-Square)"])

    with tab1:
        st.subheader("Odds Ratio (OR) Analysis of Actionable Workplace Thresholds")
        st.caption("An Odds Ratio > 1 indicates elevated odds of departure when exposed to this condition.")
        df_or = analyzer.compute_odds_ratios()

        fig_or = px.bar(
            df_or,
            x="odds_ratio",
            y="risk_factor",
            orientation="h",
            color="odds_ratio",
            color_continuous_scale="Purples",
            title="Odds Ratio of Leaving Company (95% CI)",
        )
        fig_or.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_or, use_container_width=True)
        st.dataframe(df_or, use_container_width=True)

    with tab2:
        st.subheader("Comparison of Metric Means: Attrited vs. Retained Cohorts")
        num_cols = [
            "overtime_hours_month", "compa_ratio", "weekly_hours", "job_satisfaction",
            "work_life_balance", "monthly_income", "tenure_months", "months_since_last_promotion",
            "training_hours_completed", "sick_leave_days"
        ]
        stat_num = analyzer.continuous_feature_tests(num_cols)
        st.dataframe(
            stat_num.style.format({
                "mean_attrited": "{:.2f}",
                "mean_retained": "{:.2f}",
                "absolute_diff": "{:+.2f}",
                "pct_difference": "{:+.1f}%",
                "t_statistic": "{:.3f}",
                "p_value": "{:.2e}",
                "mann_whitney_p": "{:.2e}",
                "correlation": "{:+.3f}",
            }),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Categorical Drivers (Chi-Square Test of Independence)")
        cat_cols = ["department", "job_role", "contract_type", "education_level"]
        stat_cat = analyzer.categorical_chi2_tests(cat_cols)
        st.dataframe(stat_cat, use_container_width=True)


# ---------------- PAGE 3: INDIVIDUAL RISK & PLAYBOOKS ----------------
elif nav == "Individual Risk & Playbooks":
    st.title("🎯 Individual Employee Diagnostic & Retention Playbooks")
    st.markdown("Inspect any specific employee to examine their multi-factor root drivers and recommended interventions.")

    # Select employee
    emp_ids = filtered_df["employee_id"].tolist()
    selected_emp_id = st.selectbox("Select Employee Identifier", options=emp_ids)
    emp_row = filtered_df[filtered_df["employee_id"] == selected_emp_id].iloc[0]

    rec = recommender.generate_individual_recommendations(emp_row, predicted_risk=emp_row["predicted_risk"])

    c1, c2, c3, c4 = st.columns(4)
    badge_class = f"badge-{rec['risk_level'].lower()}"
    c1.markdown(f"**Flight Risk Score:**<br><span style='font-size:1.8rem; font-weight:700;'>{rec['risk_score']:.1%}</span>", unsafe_allow_html=True)
    c2.markdown(f"**Risk Level:**<br><span class='{badge_class}'>{rec['risk_level']}</span>", unsafe_allow_html=True)
    c3.metric("Cost to Replace", f"${rec['estimated_replacement_cost_avoidance']:,.0f}")
    c4.metric("Est. Intervention Cost", f"${rec['estimated_intervention_cost']:,}")

    st.markdown("---")

    col_details, col_actions = st.columns([1, 1.2])

    with col_details:
        st.subheader("Employee Profile & Work Metrics")
        st.write(f"**Department:** {emp_row['department']} | **Role:** {emp_row['job_role']}")
        st.write(f"**Contract:** {emp_row['contract_type']} | **Tenure:** {emp_row['tenure_months']} months")
        st.write(f"**Monthly Income:** ${emp_row['monthly_income']:,} | **Compa-Ratio:** {emp_row['compa_ratio']:.2f}")
        st.write(f"**Monthly Overtime:** {emp_row['overtime_hours_month']} hrs | **Weekly Hours:** {emp_row['weekly_hours']} hrs")
        st.write(f"**Job Satisfaction:** {emp_row['job_satisfaction']}/5 | **Work-Life Balance:** {emp_row['work_life_balance']}/5")
        st.write(f"**Performance Rating:** {emp_row['performance_rating']}/5 | **Training Completed:** {emp_row['training_hours_completed']} hrs")

        st.subheader("Identified Attrition Root Causes")
        if rec["primary_drivers"]:
            for d in rec["primary_drivers"]:
                st.error(f"⚠️ {d}")
        else:
            st.success("✅ No primary high-risk drivers triggered.")

    with col_actions:
        st.subheader("Tailored Intervention Action Playbook")
        st.markdown(f"**Projected ROI of Retention Action:** **{rec['projected_intervention_roi']:.1f}x**")

        for idx, inter in enumerate(rec["recommended_interventions"], 1):
            with st.expander(f"{idx}. {inter['title']} ({inter['category']})", expanded=True):
                st.markdown(f"**Expected Impact:** {inter['impact']}")
                st.markdown(f"**Timeline:** ~{inter['estimated_implementation_days']} days | **Cost:** ${inter['cost_per_employee']}")
                st.markdown("**Actionable Steps:**")
                for act in inter["actions"]:
                    st.write(f"- {act}")


# ---------------- PAGE 4: WHAT-IF POLICY SIMULATOR ----------------
elif nav == "What-If Policy Simulator":
    st.title("🧪 What-If Organizational Policy Intervention Simulator")
    st.markdown("Simulate enterprise-wide policy interventions (e.g., overtime caps, compensation corrections) and calculate projected retention gains.")

    st.subheader("Select Hypothetical HR Interventions to Simulate:")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        cap_overtime = st.checkbox("Policy A: Institute Strict Overtime Cap (Max 10 hrs/month)", value=True)
        compa_floor = st.checkbox("Policy B: Bring all sub-benchmark salaries to 0.95 Compa-Ratio", value=True)

    with sim_col2:
        career_track = st.checkbox("Policy C: Upskilling & Career Progression for Stagnant Staff (>2 yrs)", value=False)
        remote_boost = st.checkbox("Policy D: 50% Remote Flexibility for Commuters (>35km)", value=False)

    if predictor is not None and preprocessor is not None:
        sim_df = filtered_df.copy()

        # Apply simulated interventions
        if cap_overtime:
            sim_df["overtime_hours_month"] = np.minimum(sim_df["overtime_hours_month"], 10.0)
            sim_df["weekly_hours"] = np.minimum(sim_df["weekly_hours"], 42.5)
            sim_df["work_life_balance"] = np.clip(sim_df["work_life_balance"] + 1, 1, 5)

        if compa_floor:
            sim_df["compa_ratio"] = np.maximum(sim_df["compa_ratio"], 0.95)

        if career_track:
            mask = sim_df["months_since_last_promotion"] > 24
            sim_df.loc[mask, "career_growth_opportunities"] = np.clip(sim_df.loc[mask, "career_growth_opportunities"] + 1, 1, 5)
            sim_df.loc[mask, "training_hours_completed"] += 15

        if remote_boost:
            mask_commute = sim_df["commute_distance_km"] > 35
            sim_df.loc[mask_commute, "remote_work_pct"] = np.maximum(sim_df.loc[mask_commute, "remote_work_pct"], 50)

        sim_trans = preprocessor.transform(sim_df)
        sim_df["simulated_risk"] = predictor.predict_proba(sim_trans)

        baseline_risk = filtered_df["predicted_risk"].mean()
        sim_risk = sim_df["simulated_risk"].mean()
        risk_reduction = baseline_risk - sim_risk

        baseline_high_risk = (filtered_df["predicted_risk"] >= 0.5).sum()
        sim_high_risk = (sim_df["simulated_risk"] >= 0.5).sum()
        employees_saved = baseline_high_risk - sim_high_risk

        avg_salary = filtered_df["monthly_income"].mean() * 12
        saved_turnover_dollars = employees_saved * (avg_salary * 0.5)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Mean Risk", f"{baseline_risk:.1%}")
        m2.metric("Simulated Risk", f"{sim_risk:.1%}", f"-{risk_reduction:.1%}", delta_color="inverse")
        m3.metric("At-Risk Staff Saved", f"{employees_saved:,} people")
        m4.metric("Est. Turnover Cost Saved", f"${saved_turnover_dollars:,.0f}")

        # Distribution comparison
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=filtered_df["predicted_risk"], name="Baseline Risk", opacity=0.6, marker_color="#ef4444"))
        fig_sim.add_trace(go.Histogram(x=sim_df["simulated_risk"], name="Simulated Policy Risk", opacity=0.6, marker_color="#10b981"))
        fig_sim.update_layout(barmode="overlay", title="Attrition Risk Probability Density: Before vs. After Policy Interventions")
        fig_sim.update_xaxes(title_text="Risk Probability")
        fig_sim.update_yaxes(title_text="Employee Count")
        st.plotly_chart(fig_sim, use_container_width=True)


# ---------------- PAGE 5: COHORT & SEASONAL INTELLIGENCE ----------------
elif nav == "Cohort & Seasonal Intelligence":
    st.title("📅 Cohort & Seasonal Retention Intelligence")
    st.markdown("Specific analytics for managing seasonal employee contracts, turnover during peak spikes, and onboarding tenure cohorts.")

    cohorts = analyzer.identify_high_risk_cohorts()
    st.subheader("Identified High-Risk Workforce Segments")
    st.dataframe(pd.DataFrame(cohorts), use_container_width=True)

    st.subheader("Contract Type & Seasonal Pressure Impact")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        contract_summary = (
            filtered_df.groupby(["contract_type", "peak_season_pressure"])["predicted_risk"]
            .mean()
            .reset_index()
        )
        contract_summary["peak_season_pressure"] = contract_summary["peak_season_pressure"].map({1: "Peak Season Spike", 0: "Normal Period"})

        fig_contract = px.bar(
            contract_summary,
            x="contract_type",
            y="predicted_risk",
            color="peak_season_pressure",
            barmode="group",
            title="Attrition Risk: Contract Types under Peak Season Strain",
        )
        fig_contract.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_contract, use_container_width=True)

    with col_c2:
        st.subheader("Tenure Trajectory & Early Onboarding Vulnerability")
        filtered_df["tenure_band"] = pd.cut(
            filtered_df["tenure_months"],
            bins=[0, 6, 12, 24, 48, 150],
            labels=["0-6 Months (Onboarding)", "7-12 Months (Early)", "1-2 Years", "2-4 Years", "4+ Years"],
        )
        tenure_summary = filtered_df.groupby("tenure_band")["predicted_risk"].mean().reset_index()

        fig_tenure = px.line(
            tenure_summary,
            x="tenure_band",
            y="predicted_risk",
            markers=True,
            title="Attrition Risk Curve across Employee Tenure Milestones",
        )
        fig_tenure.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.subheader("Seasonal Contract Mitigation Framework")
    st.info("""
    **Playbook for Seasonal & Peak Season Retention:**
    1. **Pre-Peak Contract Security:** Announce performance-based full-time transition criteria 30 days prior to peak season start.
    2. **Retention Milestone Bonus:** Structure 20% completion bonus distributed upon completion of peak period.
    3. **Shift Load Leveling:** Avoid back-to-back 50+ hour work weeks for seasonal staff; supplement with temporary surge labor.
    4. **Onboarding Buddy Assignment:** Pair seasonal recruits with 2nd-year tenured veterans to mitigate day-1 to day-30 ramp-up disengagement.
    """)

