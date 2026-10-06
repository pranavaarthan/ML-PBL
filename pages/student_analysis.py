"""
Student Detail Analysis Page - Individual Behavioral Profile, Gauges, Risk, and Explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    apply_theme,
    get_risk_badge_html,
    get_engagement_badge_html
)
from src.explainability import explain_student_prediction


def render_student_analysis_page():
    apply_theme()

    st.markdown("## 👨‍🎓 Individual Student Behavioral & Academic Analysis")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -10px;'>"
        "Granular diagnostics, ML prediction explainability, and personalized intervention planning."
        "</p>",
        unsafe_allow_html=True
    )

    if "df" not in st.session_state or st.session_state["df"] is None:
        pass
        return

    df = st.session_state["df"]

    # Student selector & search
    student_ids = list(df["Student_ID"].dropna().unique())

    # Pre-select if redirected from another page
    default_idx = 0
    if "selected_student_id" in st.session_state and st.session_state["selected_student_id"] in student_ids:
        default_idx = student_ids.index(st.session_state["selected_student_id"])

    col_search1, col_search2 = st.columns([2, 1])
    with col_search1:
        selected_id = st.selectbox(
            "Select or Search Student ID:",
            options=student_ids,
            index=default_idx,
            help="Type to quickly locate a specific student"
        )
        st.session_state["selected_student_id"] = selected_id

    # Retrieve student record
    student_records = df[df["Student_ID"] == selected_id]
    if student_records.empty:
        pass
        return

    student = student_records.iloc[0].to_dict()

    # ========================================================
    # STUDENT PROFILE CARD
    # ========================================================
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    eng_level = student.get("Predicted_Engagement_Level") or student.get("Engagement_Level", "Unknown")
    risk_level = student.get("Risk_Level", "Low")
    risk_badge = get_risk_badge_html(risk_level)
    eng_badge = get_engagement_badge_html(eng_level)

    profile_card_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h3 style="margin: 0; color: #0f172a; font-size: 1.4rem;">Student Profile: <span style="color: #3b82f6;">{selected_id}</span></h3>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.9rem;">
                    Delivery Mode: <strong>{student.get('Delivery_Method', 'N/A')}</strong> &nbsp;|&nbsp; 
                    Logged Emotion: <strong>{student.get('Primary_Emotion', 'N/A')}</strong>
                </p>
            </div>
            <div style="margin-top: 8px;">
                {eng_badge} &nbsp; {risk_badge}
            </div>
        </div>
    </div>
    """
    st.markdown(profile_card_html, unsafe_allow_html=True)

    # ========================================================
    # INTERACTIVE GAUGES & VISUAL INDICATORS
    # ========================================================
    g1, g2, g3 = st.columns(3)

    # Gauge 1: Engagement Score Gauge
    with g1:
        eng_score = float(student.get("Engagement_Score", 0.0))
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eng_score,
            title={"text": "Engagement Score", "font": {"size": 15, "color": "#1e293b"}},
            number={"valueformat": ".3f", "font": {"size": 24, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0.0, 0.25], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0.0, 0.10], "color": "#fee2e2"},
                    {"range": [0.10, 0.14], "color": "#eff6ff"},
                    {"range": [0.14, 0.25], "color": "#d1fae5"}
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 3},
                    "thickness": 0.75,
                    "value": 0.10
                }
            }
        ))
        fig_g1.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=230)
        st.plotly_chart(fig_g1, use_container_width=True)

    # Gauge 2: Academic Success Gauge
    with g2:
        pred_acad = float(student.get("Predicted_Academic_Score", student.get("Academic_Success_Score", 0.0)))
        actual_acad = student.get("Academic_Success_Score")

        delta_config = None
        if actual_acad is not None:
            actual_float = float(actual_acad)
            delta_val = pred_acad - actual_float
            delta_config = {"reference": actual_float, "valueformat": "+.1f", "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}}

        mode_str = "gauge+number+delta" if delta_config else "gauge+number"
        fig_g2 = go.Figure(go.Indicator(
            mode=mode_str,
            value=pred_acad,
            delta=delta_config,
            title={"text": "Academic Success (%)", "font": {"size": 15, "color": "#1e293b"}},
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 24, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0.0, 100.0], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0.0, 55.0], "color": "#fee2e2"},
                    {"range": [55.0, 75.0], "color": "#fef3c7"},
                    {"range": [75.0, 100.0], "color": "#d1fae5"}
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 3},
                    "thickness": 0.75,
                    "value": 55.0
                }
            }
        ))
        fig_g2.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=230)
        st.plotly_chart(fig_g2, use_container_width=True)

    # Indicator 3: Weekly Clicks Activity vs Cohort Mean
    with g3:
        clicks_val = int(student.get("Weekly_Clicks", 0))
        cohort_mean_clicks = df["Weekly_Clicks"].mean()
        fig_g3 = go.Figure(go.Indicator(
            mode="number+delta",
            value=clicks_val,
            title={"text": "Weekly LMS Clicks", "font": {"size": 15, "color": "#1e293b"}},
            number={"font": {"size": 32, "color": "#0f172a"}},
            delta={
                "reference": cohort_mean_clicks,
                "position": "bottom",
                "valueformat": "+.1f",
                "increasing": {"color": "#10b981"},
                "decreasing": {"color": "#ef4444"}
            }
        ))
        fig_g3.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=230,
            template="plotly_white"
        )
        st.plotly_chart(fig_g3, use_container_width=True)

    # ========================================================
    # RISK ASSESSMENT & RECOMMENDED INTERVENTION
    # ========================================================
    st.markdown("### ⚠️ Risk Diagnostics & Recommended Intervention")

    risk_col1, risk_col2 = st.columns([1, 1])

    with risk_col1:
        reasons_raw = student.get("Risk_Reasons", "")
        # Format reasons cleanly
        reasons_items = [r.strip() for r in str(reasons_raw).split("•") if r.strip()]
        if not reasons_items:
            reasons_items = ["Engagement metrics and academic projections remain within optimal thresholds."]

        st.markdown(f"**Assessed Risk Category: {risk_badge}**", unsafe_allow_html=True)
        st.markdown("**Contributing Signals & Risk Factors:**")
        for item in reasons_items:
            st.markdown(f"- {item}")

    with risk_col2:
        intervention_text = student.get(
            "Recommended_Intervention",
            "Maintain standard monitoring and supportive instructional feedback."
        )
        st.markdown("**Actionable Faculty Intervention Recommendation:**")
        st.info(f"📋 **Action Plan:**\n\n{intervention_text}")

    st.markdown("---")

    # ========================================================
    # MODEL EXPLAINABILITY (SHAP & FEATURE ATTRIBUTIONS)
    # ========================================================
    st.markdown("### 🔍 Explainable AI: Why Was This Student Classified This Way?")

    clf_model = st.session_state.get("clf_model")
    if clf_model is None:
        pass
        return

    student_df = pd.DataFrame([student])

    with st.spinner("Computing Shapley feature attributions..."):
        try:
            explanation = explain_student_prediction(clf_model, student_df)

            col_exp1, col_exp2 = st.columns([1, 1])

            with col_exp1:
                st.markdown(f"**Predicted Engagement:** `{explanation['predicted_class']}`")
                st.markdown(f"**Attribution Methodology:** `{explanation['method']}`")
                st.markdown("**Ranked Contributing Features:**")

                for factor in explanation["top_factors"]:
                    icon = "🟢" if factor["impact"] >= 0 else "🔴"
                    st.markdown(
                        f"{factor['rank']}. {icon} **{factor['feature']}**: "
                        f"`{factor['impact']:+.4f}` ({factor['direction']})"
                    )

                st.caption(
                    "Positive values drive classification toward the predicted engagement level, "
                    "while negative values suppress classification confidence."
                )

            with col_exp2:
                st.plotly_chart(explanation["figure"], use_container_width=True)

        except Exception as e:
            pass

    # ========================================================
    # COHORT BENCHMARK COMPARISON
    # ========================================================
    st.markdown("### 📊 Cohort Relative Standing")
    deliv = student.get("Delivery_Method", "Online")
    cohort_df = df[df["Delivery_Method"] == deliv]

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric(
            label="Weekly Clicks vs Delivery Peers",
            value=f"{clicks_val}",
            delta=f"{clicks_val - cohort_df['Weekly_Clicks'].mean():.1f} vs {deliv} mean"
        )
    with b2:
        st.metric(
            label="Engagement Score vs Peers",
            value=f"{eng_score:.3f}",
            delta=f"{eng_score - cohort_df['Engagement_Score'].mean():.3f} vs {deliv} mean"
        )
    with b3:
        if actual_acad is not None:
            actual_fl = float(actual_acad)
            st.metric(
                label="Academic Success vs Peers",
                value=f"{actual_fl:.1f}%",
                delta=f"{actual_fl - cohort_df['Academic_Success_Score'].mean():.1f}% vs {deliv} mean"
            )
        else:
            st.metric(
                label="Predicted Academic Success vs Peers",
                value=f"{pred_acad:.1f}%",
                delta=f"{pred_acad - cohort_df['Predicted_Academic_Score'].mean():.1f}% vs {deliv} mean"
            )


if __name__ == "__main__" or True:
    render_student_analysis_page()
