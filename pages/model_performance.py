"""
Model Performance Page - Comparative Evaluation, Confusion Matrices, and Retraining Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import apply_theme, render_kpi_card
from src.model_training import train_and_evaluate_all
from src.prediction import predict_batch, load_models
from src.risk_detection import assess_batch_risk


def render_model_performance_page():
    apply_theme()

    st.markdown("## 📈 Machine Learning Model Performance & Benchmarking")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -10px;'>"
        "Comparative evaluation between Random Forest and XGBoost on real test partition. "
        "Dynamic metrics with confusion matrices and feature importance."
        "</p>",
        unsafe_allow_html=True
    )

    if "df" not in st.session_state or st.session_state["df"] is None:
        pass
        return

    df = st.session_state["df"]

    # ========================================================
    # RETRAIN MODELS WORKFLOW
    # ========================================================
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown("### 🔄 Production Models Pipeline")
    with col_head2:
        if st.button("🚀 Retrain Models", type="primary", use_container_width=True):
            with st.spinner("Retraining Random Forest & XGBoost pipelines on active dataset..."):
                try:
                    # Train and evaluate
                    new_metrics = train_and_evaluate_all(df, save_models=True)
                    st.session_state["model_metrics"] = new_metrics
                    
                    # Reload new models into session
                    clf_model, reg_model = load_models()
                    st.session_state["clf_model"] = clf_model
                    st.session_state["reg_model"] = reg_model

                    # Re-score dataset
                    enriched_df = predict_batch(clf_model, reg_model, df)
                    enriched_df = assess_batch_risk(enriched_df)
                    st.session_state["df"] = enriched_df

                    st.success("✅ Models retrained, evaluated, and saved to disk successfully!")
                    st.rerun()
                except Exception as e:
                    pass

    # Ensure evaluation metrics are available in session
    if "model_metrics" not in st.session_state or st.session_state["model_metrics"] is None:
        with st.spinner("Evaluating models on 80/20 test split to compute dynamic metrics..."):
            try:
                metrics = train_and_evaluate_all(df, save_models=False)
                st.session_state["model_metrics"] = metrics
            except Exception as e:
                pass
            return

    metrics = st.session_state["model_metrics"]
    ds_sum = metrics["dataset_summary"]
    rf_clf = metrics["classification"]["random_forest"]
    xgb_clf = metrics["classification"]["xgboost"]
    rf_reg = metrics["regression"]["random_forest"]
    xgb_reg = metrics["regression"]["xgboost"]
    classes = ds_sum["classes"]

    # Dataset partition KPI cards
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        render_kpi_card(
            label="Total Dataset Samples",
            value=f"{ds_sum['total_samples']:,}",
            subtext="Enrolled student records",
            accent_color="#6366f1"
        )
    with kp2:
        render_kpi_card(
            label="Training Partition",
            value=f"{ds_sum['train_samples']:,}",
            subtext="80% Stratified split",
            accent_color="#3b82f6"
        )
    with kp3:
        render_kpi_card(
            label="Testing Partition",
            value=f"{ds_sum['test_samples']:,}",
            subtext="20% Unseen test evaluation",
            accent_color="#8b5cf6"
        )
    with kp4:
        rf_acc_pct = rf_clf["accuracy"] * 100
        render_kpi_card(
            label="Primary RF Accuracy",
            value=f"{rf_acc_pct:.1f}%",
            subtext="Dynamic test accuracy",
            accent_color="#10b981"
        )

    st.markdown("---")

    # ========================================================
    # 1. ENGAGEMENT CLASSIFICATION COMPARISON
    # ========================================================
    st.markdown("### 🎯 1. Engagement Level Classification Benchmarking")

    comp_df = pd.DataFrame({
        "Evaluation Metric": [
            "Test Accuracy",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1-Score (Weighted)"
        ],
        "Random Forest Classifier (Primary)": [
            f"{rf_clf['accuracy'] * 100:.2f}%",
            f"{rf_clf['precision'] * 100:.2f}%",
            f"{rf_clf['recall'] * 100:.2f}%",
            f"{rf_clf['f1_score'] * 100:.2f}%"
        ],
        "XGBoost Classifier (Benchmark)": [
            f"{xgb_clf['accuracy'] * 100:.2f}%",
            f"{xgb_clf['precision'] * 100:.2f}%",
            f"{xgb_clf['recall'] * 100:.2f}%",
            f"{xgb_clf['f1_score'] * 100:.2f}%"
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Confusion Matrices
    st.markdown("#### Confusion Matrices (20% Test Partition)")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        fig_rf_cm = px.imshow(
            rf_clf["confusion_matrix"],
            labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Random Forest (Accuracy: {rf_clf['accuracy'] * 100:.1f}%)"
        )
        fig_rf_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig_rf_cm, use_container_width=True)

    with cm_col2:
        fig_xgb_cm = px.imshow(
            xgb_clf["confusion_matrix"],
            labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Greens",
            title=f"XGBoost (Accuracy: {xgb_clf['accuracy'] * 100:.1f}%)"
        )
        fig_xgb_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig_xgb_cm, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # 2. ACADEMIC SUCCESS REGRESSION COMPARISON
    # ========================================================
    st.markdown("### 📐 2. Academic Success Regression Benchmarking")

    reg_comp_df = pd.DataFrame({
        "Regression Metric": [
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "R-squared Score (R²)"
        ],
        "Random Forest Regressor (Primary)": [
            f"{rf_reg['mae']:.3f} points",
            f"{rf_reg['mse']:.3f}",
            f"{rf_reg['rmse']:.3f} points",
            f"{rf_reg['r2']:.4f}"
        ],
        "XGBoost Regressor (Benchmark)": [
            f"{xgb_reg['mae']:.3f} points",
            f"{xgb_reg['mse']:.3f}",
            f"{xgb_reg['rmse']:.3f} points",
            f"{xgb_reg['r2']:.4f}"
        ]
    })
    st.dataframe(reg_comp_df, use_container_width=True, hide_index=True)

    # Actual vs Predicted Scatter
    if "Academic_Success_Score" in df.columns and "Predicted_Academic_Score" in df.columns:
        fig_reg_scatter = px.scatter(
            df.sample(min(500, len(df)), random_state=42),
            x="Academic_Success_Score",
            y="Predicted_Academic_Score",
            color="Engagement_Level",
            color_discrete_map={"Highly Engaged": "#10b981", "Engaged": "#3b82f6", "Disengaged": "#ef4444"},
            title="Actual vs Predicted Academic Success Score (Sample N=500)",
            opacity=0.65
        )
        # 45-degree reference line
        fig_reg_scatter.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode="lines",
                line=dict(color="#94a3b8", dash="dash"),
                name="Perfect Prediction (y=x)"
            )
        )
        fig_reg_scatter.update_layout(
            xaxis_title="Actual Academic Success Score (%)",
            yaxis_title="Predicted Academic Success Score (%)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_reg_scatter, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # 3. GLOBAL FEATURE IMPORTANCES
    # ========================================================
    st.markdown("### 🌟 3. Global Model Feature Importance")
    feat_imps = metrics.get("feature_importances", {})
    if feat_imps:
        feat_df = (
            pd.DataFrame(list(feat_imps.items()), columns=["Feature", "Importance"])
            .sort_values(by="Importance", ascending=True)
        )
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Random Forest Feature Importance Weights"
        )
        fig_feat.update_layout(
            xaxis_title="Relative Gini Importance",
            yaxis_title="Feature Identifier",
            margin=dict(l=20, r=20, t=40, b=20),
            height=340
        )
        st.plotly_chart(fig_feat, use_container_width=True)


if __name__ == "__main__" or True:
    render_model_performance_page()
