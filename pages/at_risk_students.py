"""
At-Risk Students Triage and Intervention Management Page
"""

import streamlit as st
import pandas as pd
from utils.helpers import apply_theme, render_kpi_card


def render_at_risk_page():
    apply_theme()

    st.markdown("## ⚠️ At-Risk Student Intervention Roster")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -10px;'>"
        "Early warning detection system prioritizing students requiring immediate academic guidance and instructional support."
        "</p>",
        unsafe_allow_html=True
    )

    if "df" not in st.session_state or st.session_state["df"] is None:
        pass
        return

    df = st.session_state["df"].copy()

    # ========================================================
    # TOP TRIAGE METRICS
    # ========================================================
    total_cohort = len(df)
    high_risk_total = (df["Risk_Level"] == "High").sum()
    med_risk_total = (df["Risk_Level"] == "Medium").sum()
    low_risk_total = (df["Risk_Level"] == "Low").sum()
    pct_flagged = ((high_risk_total + med_risk_total) / total_cohort * 100) if total_cohort > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card(
            label="High Risk (Critical)",
            value=f"{high_risk_total:,}",
            subtext="Immediate intervention needed",
            accent_color="#dc2626"
        )
    with m2:
        render_kpi_card(
            label="Medium Risk (Advisory)",
            value=f"{med_risk_total:,}",
            subtext="Targeted monitoring required",
            accent_color="#f59e0b"
        )
    with m3:
        render_kpi_card(
            label="Low Risk (Healthy)",
            value=f"{low_risk_total:,}",
            subtext="Progressing satisfactorily",
            accent_color="#10b981"
        )
    with m4:
        render_kpi_card(
            label="Cohort Intervention Rate",
            value=f"{pct_flagged:.1f}%",
            subtext=f"{high_risk_total + med_risk_total} of {total_cohort} students",
            accent_color="#6366f1"
        )

    st.markdown("---")

    # ========================================================
    # FILTERS & SEARCH
    # ========================================================
    st.markdown("### 🔍 Filter and Triage Roster")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

    with f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
            help="Defaults to students requiring intervention"
        )

    with f2:
        available_eng = sorted(list(df["Engagement_Level"].dropna().unique()))
        eng_filter = st.multiselect(
            "Filter by Engagement Level:",
            options=available_eng,
            default=available_eng
        )

    with f3:
        available_deliv = sorted(list(df["Delivery_Method"].dropna().unique()))
        deliv_filter = st.multiselect(
            "Filter by Delivery Method:",
            options=available_deliv,
            default=available_deliv
        )

    with f4:
        sort_by = st.selectbox(
            "Sort Order:",
            options=[
                "Risk Severity (High -> Low)",
                "Academic Score (Lowest First)",
                "Weekly Clicks (Lowest First)",
                "Engagement Score (Lowest First)"
            ],
            index=0
        )

    # Filter dataframe
    roster_df = df.copy()

    if risk_filter:
        roster_df = roster_df[roster_df["Risk_Level"].isin(risk_filter)]
    if eng_filter:
        roster_df = roster_df[roster_df["Engagement_Level"].isin(eng_filter)]
    if deliv_filter:
        roster_df = roster_df[roster_df["Delivery_Method"].isin(deliv_filter)]

    # Sorting logic
    if sort_by == "Risk Severity (High -> Low)":
        risk_rank = {"High": 1, "Medium": 2, "Low": 3}
        roster_df["_sort_risk"] = roster_df["Risk_Level"].map(risk_rank).fillna(4)
        roster_df = roster_df.sort_values(by=["_sort_risk", "Engagement_Score"], ascending=[True, True])
        roster_df = roster_df.drop(columns=["_sort_risk"])
    elif sort_by == "Academic Score (Lowest First)":
        col_sort = "Academic_Success_Score" if "Academic_Success_Score" in roster_df.columns else "Predicted_Academic_Score"
        roster_df = roster_df.sort_values(by=col_sort, ascending=True)
    elif sort_by == "Weekly Clicks (Lowest First)":
        roster_df = roster_df.sort_values(by="Weekly_Clicks", ascending=True)
    elif sort_by == "Engagement Score (Lowest First)":
        roster_df = roster_df.sort_values(by="Engagement_Score", ascending=True)

    st.markdown(f"**Showing {len(roster_df):,} matching students in triage roster:**")

    # ========================================================
    # STUDENT SELECTION ACTION BAR
    # ========================================================
    action_col1, action_col2 = st.columns([2, 1])

    with action_col1:
        if not roster_df.empty:
            inspect_candidate = st.selectbox(
                "Select a student from the filtered list to examine:",
                options=roster_df["Student_ID"].tolist(),
                key="select_inspect_student"
            )
            if st.button("🔎 Open Detailed Student Analysis", type="primary"):
                st.session_state["selected_student_id"] = inspect_candidate
                try:
                    st.switch_page("pages/student_analysis.py")
                except Exception:
                    st.success(f"Selected {inspect_candidate}. Please navigate to 'Student Analysis' in the sidebar.")

    with action_col2:
        # CSV Export
        csv_data = roster_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Roster (CSV)",
            data=csv_data,
            file_name="at_risk_students_roster.csv",
            mime="text/csv",
            help="Download filtered list for faculty review and advisory meetings"
        )

    # ========================================================
    # DEDICATED TABLE
    # ========================================================
    # Select display columns
    display_cols = [
        "Student_ID",
        "Risk_Level",
        "Engagement_Level",
        "Engagement_Score",
        "Weekly_Clicks",
        "Academic_Success_Score",
        "Predicted_Academic_Score",
        "Delivery_Method",
        "Primary_Emotion",
        "Risk_Reasons"
    ]
    # Filter to existing columns
    present_cols = [c for c in display_cols if c in roster_df.columns]
    table_view = roster_df[present_cols].copy()

    # Column formatting configs
    column_config = {
        "Student_ID": st.column_config.TextColumn("Student ID", width="small"),
        "Risk_Level": st.column_config.TextColumn("Risk Level", width="small"),
        "Engagement_Level": st.column_config.TextColumn("Engagement Level", width="medium"),
        "Engagement_Score": st.column_config.NumberColumn("Engagement Score", format="%.3f"),
        "Weekly_Clicks": st.column_config.NumberColumn("Weekly Clicks", format="%d"),
        "Academic_Success_Score": st.column_config.NumberColumn("Actual Academic %", format="%.1f%%"),
        "Predicted_Academic_Score": st.column_config.NumberColumn("Predicted %", format="%.1f%%"),
        "Delivery_Method": st.column_config.TextColumn("Delivery Method", width="small"),
        "Primary_Emotion": st.column_config.TextColumn("Emotion", width="small"),
        "Risk_Reasons": st.column_config.TextColumn("Contributing Risk Signals", width="large")
    }

    st.dataframe(
        table_view,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=480
    )


if __name__ == "__main__" or True:
    render_at_risk_page()
