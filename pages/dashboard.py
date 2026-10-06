"""
Faculty Dashboard - Overview, KPIs, Dynamic Analytics, and What Changed
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    render_kpi_card,
    compute_what_changed,
    apply_theme
)


def render_dashboard_page():
    apply_theme()

    st.markdown("## 📊 Faculty Analytics Dashboard")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -10px;'>"
        "Holistic behavioral learning intelligence, cohort engagement patterns, and academic projections."
        "</p>",
        unsafe_allow_html=True
    )

    if "df" not in st.session_state or st.session_state["df"] is None:
        pass
        return

    df = st.session_state["df"].copy()

    # ========================================================
    # SIDEBAR / PAGE FILTERS
    # ========================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Cohort Filters")

    delivery_options = ["All"] + sorted(list(df["Delivery_Method"].dropna().unique()))
    selected_delivery = st.sidebar.selectbox("Delivery Method", delivery_options, index=0)

    engagement_options = ["All"] + sorted(list(df["Engagement_Level"].dropna().unique()))
    selected_engagement = st.sidebar.selectbox("Engagement Level", engagement_options, index=0)

    risk_options = ["All"] + sorted(list(df["Risk_Level"].dropna().unique()))
    selected_risk = st.sidebar.selectbox("Risk Level", risk_options, index=0)

    # Apply filters
    filtered_df = df.copy()
    if selected_delivery != "All":
        filtered_df = filtered_df[filtered_df["Delivery_Method"] == selected_delivery]
    if selected_engagement != "All":
        filtered_df = filtered_df[filtered_df["Engagement_Level"] == selected_engagement]
    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]

    # ========================================================
    # TOP KPI CARDS
    # ========================================================
    total_students = len(filtered_df)
    highly_engaged = (filtered_df["Engagement_Level"] == "Highly Engaged").sum()
    engaged = (filtered_df["Engagement_Level"] == "Engaged").sum()
    disengaged = (filtered_df["Engagement_Level"] == "Disengaged").sum()
    high_risk_count = (filtered_df["Risk_Level"] == "High").sum()
    med_risk_count = (filtered_df["Risk_Level"] == "Medium").sum()
    total_at_risk = high_risk_count + med_risk_count

    acad_col = "Academic_Success_Score" if "Academic_Success_Score" in filtered_df.columns else "Predicted_Academic_Score"
    avg_academic = filtered_df[acad_col].mean() if not filtered_df.empty else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        render_kpi_card(
            label="Total Students",
            value=f"{total_students:,}",
            subtext=f"{len(df):,} total enrolled",
            accent_color="#6366f1"
        )
    with kpi2:
        pct_he = (highly_engaged / total_students * 100) if total_students > 0 else 0
        render_kpi_card(
            label="Highly Engaged",
            value=f"{highly_engaged:,}",
            subtext=f"{pct_he:.1f}% of cohort",
            accent_color="#10b981"
        )
    with kpi3:
        pct_e = (engaged / total_students * 100) if total_students > 0 else 0
        render_kpi_card(
            label="Engaged",
            value=f"{engaged:,}",
            subtext=f"{pct_e:.1f}% of cohort",
            accent_color="#3b82f6"
        )
    with kpi4:
        pct_d = (disengaged / total_students * 100) if total_students > 0 else 0
        render_kpi_card(
            label="Disengaged",
            value=f"{disengaged:,}",
            subtext=f"{pct_d:.1f}% of cohort",
            accent_color="#ef4444"
        )
    with kpi5:
        render_kpi_card(
            label="At-Risk Students",
            value=f"{total_at_risk:,}",
            subtext=f"{high_risk_count} High, {med_risk_count} Med",
            accent_color="#dc2626"
        )
    with kpi6:
        render_kpi_card(
            label="Avg Academic Score",
            value=f"{avg_academic:.1f}%",
            subtext="Performance mean",
            accent_color="#8b5cf6"
        )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # ========================================================
    # "WHAT CHANGED?" FEATURE
    # ========================================================
    with st.expander("⏱️ **What Changed? (Period-over-Period Delta Analysis)**", expanded=False):
        col_wc1, col_wc2 = st.columns([3, 1])
        
        with col_wc2:
            st.markdown("**Comparison Source:**")
            sim_toggle = st.toggle("Simulate Previous Period Data", value=False, key="sim_prev_period")
        
        with col_wc1:
            if sim_toggle:
                # Generate realistic previous period variation for demonstration
                np.random.seed(101)
                prev_df = df.copy()
                # Simulate slightly different metrics in previous period
                prev_df["Weekly_Clicks"] = np.clip(prev_df["Weekly_Clicks"] + np.random.randint(-15, 15, size=len(prev_df)), 5, 260)
                prev_df["Engagement_Score"] = np.clip(prev_df["Engagement_Score"] + np.random.uniform(-0.02, 0.02, size=len(prev_df)), 0.01, 0.25)
                # Re-categorize a small proportion
                shift_idx = np.random.choice(prev_df.index, size=int(len(prev_df) * 0.08), replace=False)
                prev_df.loc[shift_idx, "Engagement_Level"] = np.random.choice(["Engaged", "Disengaged", "Highly Engaged"], size=len(shift_idx))
                prev_df.loc[shift_idx, "Risk_Level"] = np.random.choice(["Low", "Medium", "High"], size=len(shift_idx))
                
                what_changed = compute_what_changed(filtered_df, prev_df)
                st.markdown("#### What Changed This Week?")
                for bullet in what_changed["bullet_points"]:
                    st.markdown(f"- **{bullet}**")
            else:
                what_changed = compute_what_changed(filtered_df, None)
                st.info(f"ℹ️ {what_changed['message']}")
                st.caption("Tip: Toggle 'Simulate Previous Period Data' above or upload a historical benchmark file in Data Upload to compare period-over-period trends.")

    # ========================================================
    # INTERACTIVE PLOTLY CHARTS (8 CHARTS)
    # ========================================================
    st.markdown("### 📈 Visual Analytics & Behavioral Patterns")

    # ROW 1: Engagement Distribution (Donut) & Academic Distribution (Histogram)
    c1, c2 = st.columns(2)

    with c1:
        # Chart 1: Engagement Level Distribution
        eng_counts = filtered_df["Engagement_Level"].value_counts().reset_index()
        eng_counts.columns = ["Engagement_Level", "Count"]
        color_map_eng = {
            "Highly Engaged": "#10b981",
            "Engaged": "#3b82f6",
            "Disengaged": "#ef4444"
        }
        fig_eng = px.pie(
            eng_counts,
            names="Engagement_Level",
            values="Count",
            hole=0.45,
            title="1. Engagement Level Distribution",
            color="Engagement_Level",
            color_discrete_map=color_map_eng
        )
        fig_eng.update_traces(textinfo="percent+label", textposition="inside")
        fig_eng.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig_eng, use_container_width=True)

    with c2:
        # Chart 2: Academic Success Distribution
        fig_acad = px.histogram(
            filtered_df,
            x=acad_col,
            nbins=30,
            title="2. Academic Success Score Distribution",
            color_discrete_sequence=["#8b5cf6"],
            marginal="box"
        )
        fig_acad.add_vline(
            x=avg_academic,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Mean: {avg_academic:.1f}%",
            annotation_position="top left"
        )
        fig_acad.update_layout(
            xaxis_title="Academic Success Score (%)",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_acad, use_container_width=True)

    # ROW 2: Engagement Score Distribution & Weekly Clicks vs Academic Success
    c3, c4 = st.columns(2)

    with c3:
        # Chart 3: Engagement Score Distribution
        fig_eng_score = px.histogram(
            filtered_df,
            x="Engagement_Score",
            nbins=35,
            title="3. Engagement Score Distribution",
            color="Engagement_Level",
            color_discrete_map=color_map_eng,
            barmode="overlay",
            opacity=0.75
        )
        fig_eng_score.update_layout(
            xaxis_title="Engagement Score (0.0 - 0.25)",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_eng_score, use_container_width=True)

    with c4:
        # Chart 4: Weekly Clicks vs Academic Success
        fig_clicks_acad = px.scatter(
            filtered_df,
            x="Weekly_Clicks",
            y=acad_col,
            color="Engagement_Level",
            color_discrete_map=color_map_eng,
            hover_data=["Student_ID", "Primary_Emotion", "Delivery_Method", "Risk_Level"],
            title="4. Weekly Clicks vs Academic Success",
            opacity=0.65
        )
        fig_clicks_acad.update_layout(
            xaxis_title="Weekly LMS Clicks",
            yaxis_title="Academic Success Score (%)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_clicks_acad, use_container_width=True)

    # ROW 3: Engagement Score vs Academic Success & Delivery Method vs Engagement
    c5, c6 = st.columns(2)

    with c5:
        # Chart 5: Engagement Score vs Academic Success
        fig_eng_acad = px.scatter(
            filtered_df,
            x="Engagement_Score",
            y=acad_col,
            color="Risk_Level",
            color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#10b981"},
            hover_data=["Student_ID", "Engagement_Level", "Weekly_Clicks"],
            title="5. Engagement Score vs Academic Success (Colored by Risk)",
            opacity=0.65,
            trendline="ols"
        )
        fig_eng_acad.update_layout(
            xaxis_title="Engagement Score",
            yaxis_title="Academic Success Score (%)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_eng_acad, use_container_width=True)

    with c6:
        # Chart 6: Delivery Method vs Engagement
        deliv_eng = (
            filtered_df.groupby(["Delivery_Method", "Engagement_Level"])
            .size()
            .reset_index(name="Count")
        )
        fig_deliv = px.bar(
            deliv_eng,
            x="Delivery_Method",
            y="Count",
            color="Engagement_Level",
            barmode="group",
            color_discrete_map=color_map_eng,
            title="6. Delivery Method vs Engagement Breakdown"
        )
        fig_deliv.update_layout(
            xaxis_title="Delivery Method",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_deliv, use_container_width=True)

    # ROW 4: Primary Emotion Distribution & Risk Distribution
    c7, c8 = st.columns(2)

    with c7:
        # Chart 7: Primary Emotion Distribution
        emotion_counts = (
            filtered_df.groupby(["Primary_Emotion", "Engagement_Level"])
            .size()
            .reset_index(name="Count")
        )
        fig_emotion = px.bar(
            emotion_counts,
            x="Primary_Emotion",
            y="Count",
            color="Engagement_Level",
            color_discrete_map=color_map_eng,
            title="7. Primary Emotion Distribution & Engagement"
        )
        fig_emotion.update_layout(
            xaxis_title="Primary Emotion Logged",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_emotion, use_container_width=True)

    with c8:
        # Chart 8: Risk Distribution
        risk_counts = filtered_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        color_map_risk = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#10b981"}
        fig_risk = px.bar(
            risk_counts,
            x="Risk_Level",
            y="Count",
            color="Risk_Level",
            color_discrete_map=color_map_risk,
            title="8. Student Risk Category Distribution",
            text="Count"
        )
        fig_risk.update_layout(
            xaxis_title="Risk Category",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_risk, use_container_width=True)


if __name__ == "__main__" or True:
    render_dashboard_page()
