"""
Data Upload Page - Dataset Ingestion, Validation, Batch Scoring, and Session Sync
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import apply_theme, render_kpi_card
from src.preprocessing import validate_dataset, clean_and_impute_data, ALL_FEATURES
from src.prediction import predict_batch
from src.risk_detection import assess_batch_risk
from src.feature_engineering import add_behavioral_features


def render_data_upload_page():
    apply_theme()

    st.markdown("## 📁 Student Dataset Upload & Automated Inference")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -10px;'>"
        "Upload new student behavioral logs to validate schema, run ML predictions, and detect at-risk students."
        "</p>",
        unsafe_allow_html=True
    )

    clf_model = st.session_state.get("clf_model")
    reg_model = st.session_state.get("reg_model")

    if clf_model is None or reg_model is None:
        pass
        return

    # ========================================================
    # TEMPLATE DOWNLOAD & UPLOAD GUIDELINES
    # ========================================================
    with st.expander("ℹ️ **Dataset Requirements & Sample Template**", expanded=False):
        st.markdown(
            """
            **Required Columns:**
            - `Weekly_Clicks` *(Integer/Float >= 0)*: Total LMS interactions during the week
            - `Engagement_Score` *(Float 0.0 to 1.0)*: Quantitative engagement index
            - `Delivery_Method` *(Categorical)*: `Online`, `Hybrid`, or `In-Person`
            - `Primary_Emotion` *(Categorical)*: `Neutral`, `Happy`, `Surprised`, `Sad`, `Scared`, `Angry`, or `Disgust`

            **Optional Columns:**
            - `Student_ID` *(Unique string)*: E.g., STU_0001
            - `Engagement_Level` *(Categorical)*: True ground truth if known
            - `Academic_Success_Score` *(Float 0 to 100)*: Historical exam or GPA metric
            """
        )
        sample_template = pd.DataFrame([
            {
                "Student_ID": "STU_9001",
                "Delivery_Method": "Online",
                "Weekly_Clicks": 125,
                "Primary_Emotion": "Neutral",
                "Engagement_Score": 0.145,
                "Academic_Success_Score": 88.0
            },
            {
                "Student_ID": "STU_9002",
                "Delivery_Method": "In-Person",
                "Weekly_Clicks": 24,
                "Primary_Emotion": "Sad",
                "Engagement_Score": 0.038,
                "Academic_Success_Score": 45.0
            }
        ])
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="student_engagement_template.csv",
            mime="text/csv"
        )

    # ========================================================
    # FILE UPLOADER
    # ========================================================
    uploaded_file = st.file_uploader(
        "Choose a Student Engagement CSV file",
        type=["csv"],
        help="Upload standard comma-separated tabular files"
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to initiate data validation and inference.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        pass
        return

    # ========================================================
    # STEP 1: VALIDATION
    # ========================================================
    st.markdown("### 🔎 1. Data Schema & Quality Validation")
    val_report = validate_dataset(raw_df, require_targets=False)

    if not val_report["is_valid"]:
        return

    # Quality Summary Metrics
    q1, q2, q3 = st.columns(3)
    with q1:
        render_kpi_card(
            label="Total Records Uploaded",
            value=f"{val_report['row_count']:,}",
            subtext="Candidate student rows",
            accent_color="#3b82f6"
        )
    with q2:
        render_kpi_card(
            label="Duplicate Records",
            value=f"{val_report['duplicate_count']:,}",
            subtext="Identical rows identified",
            accent_color="#f59e0b" if val_report['duplicate_count'] > 0 else "#10b981"
        )
    with q3:
        total_miss = sum(val_report["missing_values"].values())
        render_kpi_card(
            label="Missing Values",
            value=f"{total_miss:,}",
            subtext="Imputed during preprocessing",
            accent_color="#ef4444" if total_miss > 0 else "#10b981"
        )

    # Warnings suppressed

    # ========================================================
    # DATA PREVIEW
    # ========================================================
    st.markdown("#### Uploaded Data Preview (Top 10 Rows)")
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    # ========================================================
    # RUN PREDICTIONS ACTION
    # ========================================================
    st.markdown("---")
    st.markdown("### ⚙️ 2. Automated ML Preprocessing & Dual Prediction")

    if st.button("🚀 Process & Generate Model Predictions", type="primary"):
        with st.spinner("Executing pipeline: Cleaning -> Imputing -> Classification -> Regression -> Risk Detection..."):
            try:
                # 1. Clean & Impute
                cleaned_df = clean_and_impute_data(raw_df)
                
                # 2. Behavioral Features
                featured_df = add_behavioral_features(cleaned_df)

                # 3. Model Dual Prediction
                predicted_df = predict_batch(clf_model, reg_model, featured_df)

                # 4. Multi-factor Risk Assessment
                enriched_df = assess_batch_risk(predicted_df)

                st.session_state["latest_upload_results"] = enriched_df
                st.success("✅ Dual predictions and risk assessments computed successfully!")

            except Exception as e:
                pass
                return

    # Display results if available in session
    if "latest_upload_results" in st.session_state:
        res_df = st.session_state["latest_upload_results"]

        st.markdown("#### Prediction Summary")
        ps1, ps2, ps3 = st.columns(3)
        with ps1:
            high_count = (res_df["Risk_Level"] == "High").sum()
            render_kpi_card(
                label="High Risk Identified",
                value=f"{high_count}",
                subtext="Immediate intervention candidates",
                accent_color="#dc2626"
            )
        with ps2:
            dis_count = (res_df["Predicted_Engagement_Level"] == "Disengaged").sum()
            render_kpi_card(
                label="Disengaged Predictions",
                value=f"{dis_count}",
                subtext="Low portal participation",
                accent_color="#ef4444"
            )
        with ps3:
            avg_pred = res_df["Predicted_Academic_Score"].mean()
            render_kpi_card(
                label="Average Predicted Academic",
                value=f"{avg_pred:.1f}%",
                subtext="Cohort projected mean",
                accent_color="#8b5cf6"
            )

        st.markdown("#### Scored Dataset with Risk Diagnostics")
        preview_cols = [
            "Student_ID", "Delivery_Method", "Weekly_Clicks", "Primary_Emotion",
            "Engagement_Score", "Predicted_Engagement_Level", "Engagement_Confidence",
            "Predicted_Academic_Score", "Risk_Level", "Risk_Reasons"
        ]
        available_p_cols = [c for c in preview_cols if c in res_df.columns]
        st.dataframe(res_df[available_p_cols], use_container_width=True, hide_index=True)

        # Actions: Sync to session or export
        act1, act2 = st.columns(2)
        with act1:
            if st.button("📌 Set as Active Session Dataset (Updates Dashboard & Roster)", type="primary"):
                # Ensure Engagement_Level column exists for dashboard views
                if "Engagement_Level" not in res_df.columns:
                    res_df["Engagement_Level"] = res_df["Predicted_Engagement_Level"]
                st.session_state["df"] = res_df
                st.success("Dataset successfully set as active! Open 'Dashboard' or 'At-Risk Students' to view.")

        with act2:
            scored_csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Scored Predictions (CSV)",
                data=scored_csv,
                file_name="student_engagement_predictions.csv",
                mime="text/csv"
            )


if __name__ == "__main__" or True:
    render_data_upload_page()
