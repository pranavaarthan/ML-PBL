# Intelligent Student Engagement Analytics & Behavioral Learning Intelligence Platform

A Machine Learning Project-Based Learning (PBL) System for Higher Education

---

## 📌 1. Project Overview & Problem Statement
In contemporary higher education—spanning physical lecture halls, online portals, and blended classrooms—early detection of student disengagement is vital for academic retention. Traditional evaluation methods are retrospective, often identifying struggling students only after midterm exams or course withdrawals.

The **Intelligent Student Engagement Analytics & Behavioral Learning Intelligence Platform** is an end-to-end, machine-learning-powered faculty intelligence system. It synthesizes behavioral clickstream logs, affective indicators, and delivery methods to:
1. **Classify real-time engagement levels** (*Disengaged*, *Engaged*, *Highly Engaged*).
2. **Predict academic success scores** (*0.0% to 100.0%* continuous trajectory).
3. **Detect at-risk students** using transparent multi-signal heuristics and machine learning predictions.
4. **Explain individual model decisions** using Shapley additive explanations (SHAP).
5. **Empower faculty through an interactive, multi-page analytics dashboard**.

---

## 🎯 2. Objectives
- **Proactive Early Warning**: Automatically flag students vulnerable to academic failure before exams.
- **Explainable AI (XAI)**: Demystify predictions so faculty understand *why* a student is classified into a specific tier.
- **Fairness & Transparent Risk Logic**: Ensure affective signals (emotions) never trigger at-risk status in isolation, but provide contextual nuance alongside concrete participation metrics.
- **Actionable Faculty Workflow**: Deliver targeted interventions, downloadable rosters, and period-over-period delta tracking ("What Changed?").

---

## 🏗️ 3. System Architecture & ML Pipeline

```
Raw Student Data (CSV)
         ↓
Data Ingestion & Integrity Validation (src/preprocessing.py)
         ↓
Cleaning, Range Clamping & Median/Mode Imputation
         ↓
Feature Engineering & Behavioral Indicators (src/feature_engineering.py)
         ↓
Scikit-learn ColumnTransformer (StandardScaler + OneHotEncoder)
         ↓
Dual Model Predictions (src/prediction.py)
  ├── Random Forest Classifier  → Predicted Engagement Level & Confidence
  └── Random Forest Regressor   → Predicted Academic Success Score (%)
         ↓
Transparent Risk Detection Engine (src/risk_detection.py)
  → Risk Severity (High, Medium, Low) + Contributing Signals + Action Plan
         ↓
Explainable AI Engine (src/explainability.py)
  → TreeExplainer SHAP Attributions & Feature Importance Charts
         ↓
Interactive Faculty Streamlit Interface (app.py & pages/)
  ├── 📊 Dashboard (Overview, KPIs, 8 Plotly charts, What Changed)
  ├── 👨‍🎓 Student Analysis (Gauges, Profile, SHAP explanations)
  ├── ⚠️ At-Risk Students (Triage table, filters, CSV export)
  ├── 📈 Model Performance (RF vs XGBoost benchmarking, retrain workflow)
  └── 📁 Data Upload (CSV schema validator, automated batch scoring)
```

---

## 📁 4. Project Structure

```
Folder/
├── app.py                     # Main application entry point & navigation
├── pages/                     # Multi-page dashboard modules
│   ├── dashboard.py           # 📊 Cohort overview, KPIs, 8 dynamic charts, What Changed
│   ├── student_analysis.py    # 👨‍🎓 Individual student diagnostics, gauges & SHAP
│   ├── at_risk_students.py    # ⚠️ At-risk triage table, filters, CSV export
│   ├── model_performance.py   # 📈 Model benchmarking (RF vs XGBoost) & Retrain
│   └── data_upload.py         # 📁 CSV upload, schema validation & batch scoring
├── data/
│   └── learner_engagement_dataset_3000.csv  # 3,000 student behavioral dataset
├── models/
│   ├── engagement_classifier.pkl            # Pretrained Random Forest Classifier pipeline
│   └── academic_regressor.pkl               # Pretrained Random Forest Regressor pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data loading, validation, cleaning, ColumnTransformer
│   ├── feature_engineering.py # Behavioral indicators & cohort percentile metrics
│   ├── prediction.py          # Model loading, batch inference, confidence estimation
│   ├── risk_detection.py      # Multi-factor transparent risk detection & interventions
│   ├── explainability.py      # SHAP TreeExplainer & feature attribution charts
│   └── model_training.py      # RF & XGBoost training, comparative evaluation
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Academic UI theme, KPI cards, What Changed engine
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Comprehensive unit & integration test suite
├── requirements.txt           # Minimal production dependencies
└── README.md                  # Complete project documentation
```

---

## 📊 5. Dataset Description

The system utilizes `learner_engagement_dataset_3000.csv`, containing 3,000 student behavioral records:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `Student_ID` | String | Unique student identifier (`STU_0001` to `STU_3000`) |
| `Delivery_Method` | Categorical | Course delivery modality (`Online`, `Hybrid`, `In-Person`) |
| `Weekly_Clicks` | Integer | Total weekly interaction volume in learning management system |
| `Primary_Emotion` | Categorical | Dominant facial/affective sentiment logged during sessions |
| `Engagement_Score` | Float | Calculated quantitative engagement index (`0.000` to `0.250`) |
| `Engagement_Level` | Categorical | Target class: `Disengaged`, `Engaged`, `Highly Engaged` |
| `Academic_Success_Score` | Float | Continuous academic success indicator (`0.0%` to `100.0%`) |

---

## 🤖 6. Machine Learning Methodology & Algorithms

### Engagement Classification
- **Primary Model**: `RandomForestClassifier(n_estimators=100, random_state=42)` encapsulated within a Scikit-learn `Pipeline` and `ColumnTransformer`.
- **Benchmark Model**: `xgboost.XGBClassifier(n_estimators=100, eval_metric="mlogloss")`.
- **Metrics Evaluated**:
  - Test Accuracy: **> 99.5%**
  - Weighted Precision, Recall, and F1-Score: **> 0.99**
  - Interactive multi-class Confusion Matrix

### Academic Success Score Regression
- **Primary Model**: `RandomForestRegressor(n_estimators=100, random_state=42)`.
- **Benchmark Model**: `xgboost.XGBRegressor(n_estimators=100)`.
- **Metrics Evaluated**:
  - Mean Absolute Error (MAE): **~5.5 points**
  - Root Mean Squared Error (RMSE): **~6.8 points**
  - $R^2$ Score: **~0.88**

### Explainability (XAI)
- **SHAP (SHapley Additive exPlanations)**: Uses `shap.TreeExplainer` on the fitted random forest trees to decompose individual predictions into positive and negative marginal contributions from `Engagement_Score`, `Weekly_Clicks`, `Delivery_Method`, and `Primary_Emotion`.

---

## ⚙️ 7. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Virtual environment recommended

### Installation Steps
```bash
# 1. Clone or open the project folder
cd Folder

# 2. Activate virtual environment (if using existing .venv)
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Or Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 8. Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Once running, access the platform in your browser at:
`http://localhost:8501`

---

## 🧪 9. Running Tests

Execute the automated test suite verifying all 10 core pipeline modules:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 💡 10. Dashboard User Guide

1. **📊 Dashboard**:
   - Inspect cohort KPI cards (Total Students, Disengaged, At-Risk, Average Academic Score).
   - Explore 8 interactive charts mapping behavioral patterns across delivery modes and emotional states.
   - Use the **"What Changed?"** module to track period-over-period shifts or simulate historical changes.
2. **👨‍🎓 Student Analysis**:
   - Search any `Student_ID` (e.g. `STU_0002`).
   - Read engagement and academic gauges with target thresholds.
   - Inspect the **Explainable AI (SHAP)** breakdown to understand the root drivers of the model's classification.
3. **⚠️ At-Risk Students**:
   - Filter students by Risk Level (`High`, `Medium`, `Low`), Engagement Level, or Delivery Method.
   - Review specific contributing reasons and recommended faculty interventions.
   - Select a student and jump directly into detailed analysis or export the roster to CSV.
4. **📈 Model Performance**:
   - View dynamically calculated performance metrics comparing Random Forest and XGBoost.
   - Inspect side-by-side Confusion Matrices and Feature Importance rankings.
   - Click **"Retrain Models"** to execute a full retraining run on disk with updated models.
5. **📁 Data Upload**:
   - Upload new student logs in CSV format.
   - Automatically validate column schema and data quality.
   - Generate batch predictions and sync the scored cohort to the live dashboard.

---

## 🔮 11. Future Improvements
- **LMS API Integration**: Ingest real-time Canvas/Moodle API webhooks instead of batch CSV imports.
- **Longitudinal Trend Modeling**: Time-series LSTM/GRU models tracking week-by-week student momentum.
- **Automated Advisory Mailers**: Direct email integration to dispatch personalized academic encouragement messages.
