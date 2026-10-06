"""
Intelligent Student Engagement Analytics & Behavioral Learning Intelligence Platform
College Machine Learning PBL Project
Main Entry Point
"""

import sys
import os
import streamlit as st
import pandas as pd

# Add current workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import apply_theme
from src.preprocessing import load_dataset, clean_and_impute_data
from src.feature_engineering import add_behavioral_features
from src.prediction import load_models, predict_batch
from src.risk_detection import assess_batch_risk


# Configure Streamlit page settings
st.set_page_config(
    page_title="Student Engagement Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom academic styling
apply_theme()


@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Loads and caches production ML models."""
    return load_models()


@st.cache_data(show_spinner=False)
def get_initial_dataset():
    """Loads default dataset, computes predictions and risk assessments."""
    raw_df = load_dataset()
    cleaned = clean_and_impute_data(raw_df)
    featured = add_behavioral_features(cleaned)

    # Load models to enrich baseline dataset
    try:
        clf, reg = load_models()
        predicted = predict_batch(clf, reg, featured)
    except Exception:
        predicted = featured

    assessed = assess_batch_risk(predicted)
    return assessed


# Initialize Global Session State
if "df" not in st.session_state:
    try:
        st.session_state["df"] = get_initial_dataset()
    except Exception as e:
        st.session_state["df"] = None
        pass

if "clf_model" not in st.session_state or "reg_model" not in st.session_state:
    try:
        clf_model, reg_model = get_cached_models()
        st.session_state["clf_model"] = clf_model
        st.session_state["reg_model"] = reg_model
    except Exception as e:
        st.session_state["clf_model"] = None
        st.session_state["reg_model"] = None
        pass


# Sidebar Header & Metadata
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 15px 0;">
        <h2 style="margin: 0; font-size: 1.3rem; color: #1e293b; font-weight: 700;">
            🎓 Learner Intelligence
        </h2>
        <p style="margin: 2px 0 0 0; font-size: 0.78rem; color: #64748b;">
            Behavioral ML Analytics Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Define Multi-page Navigation
dashboard_page = st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True)
student_page = st.Page("pages/student_analysis.py", title="Student Analysis", icon="👨‍🎓")
risk_page = st.Page("pages/at_risk_students.py", title="At-Risk Students", icon="⚠️")
model_page = st.Page("pages/model_performance.py", title="Model Performance", icon="📈")
upload_page = st.Page("pages/data_upload.py", title="Data Upload", icon="📁")

pg = st.navigation(
    {
        "Student Engagement Analytics": [
            dashboard_page,
            student_page,
            risk_page,
            model_page,
            upload_page
        ]
    }
)

# Sidebar System Health Status
st.sidebar.markdown("---")
st.sidebar.caption("System Status")

if st.session_state.get("df") is not None:
    n_records = len(st.session_state["df"])
    st.sidebar.success(f"● Dataset: {n_records:,} Students Loaded", icon="📊")
else:
    st.sidebar.error("● Dataset: Missing / Unloaded", icon="❌")

if st.session_state.get("clf_model") is not None and st.session_state.get("reg_model") is not None:
    st.sidebar.info("● ML Models: Dual RF Pipelines Ready", icon="🤖")
else:
    st.sidebar.warning("● ML Models: Offline", icon="⚠️")

st.sidebar.caption("College ML PBL Project v1.0")

# Run active page
pg.run()
