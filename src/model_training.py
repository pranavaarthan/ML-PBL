"""
Machine Learning Model Training, Comparison, and Evaluation Module
Supports Random Forest and XGBoost with Scikit-learn Pipelines.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.preprocessing import (
    get_preprocessor,
    prepare_data,
    clean_and_impute_data,
    ALL_FEATURES
)


CLASS_MAPPING = {
    "Disengaged": 0,
    "Engaged": 1,
    "Highly Engaged": 2
}
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def train_and_evaluate_all(
    df: pd.DataFrame,
    save_models: bool = False
) -> Dict[str, Any]:
    """
    Trains and compares Random Forest and XGBoost for both Engagement Classification
    and Academic Success Regression.
    
    Dynamically computes real performance metrics on the test partition.
    """
    df_clean = clean_and_impute_data(df)
    X, y_class, y_reg = prepare_data(df_clean)

    if y_class is None or y_reg is None:
        raise ValueError("Dataset must contain target columns 'Engagement_Level' and 'Academic_Success_Score' to train models.")

    # 80/20 Stratified train-test split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    n_train = len(X_train)
    n_test = len(X_test)
    classes_list = sorted(list(y_class.unique()))

    # ========================================================
    # 1. RANDOM FOREST CLASSIFIER
    # ========================================================
    rf_clf_pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_clf_pipeline.fit(X_train, y_class_train)
    rf_class_preds = rf_clf_pipeline.predict(X_test)

    rf_acc = float(accuracy_score(y_class_test, rf_class_preds))
    rf_prec = float(precision_score(y_class_test, rf_class_preds, average="weighted", zero_division=0))
    rf_rec = float(recall_score(y_class_test, rf_class_preds, average="weighted", zero_division=0))
    rf_f1 = float(f1_score(y_class_test, rf_class_preds, average="weighted", zero_division=0))
    rf_cm = confusion_matrix(y_class_test, rf_class_preds, labels=classes_list).tolist()
    rf_report = classification_report(y_class_test, rf_class_preds, output_dict=True, zero_division=0)

    # Feature Importance for Random Forest Classifier
    feat_names = list(rf_clf_pipeline.named_steps["preprocessor"].get_feature_names_out())
    rf_importances = rf_clf_pipeline.named_steps["classifier"].feature_importances_
    feat_imp_dict = {
        name.replace("num__", "").replace("cat__", ""): float(val)
        for name, val in zip(feat_names, rf_importances)
    }

    # ========================================================
    # 2. XGBOOST CLASSIFIER
    # ========================================================
    y_class_train_num = y_class_train.map(CLASS_MAPPING)
    y_class_test_num = y_class_test.map(CLASS_MAPPING)

    xgb_clf_pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss"))
    ])
    xgb_clf_pipeline.fit(X_train, y_class_train_num)
    xgb_class_preds_num = xgb_clf_pipeline.predict(X_test)
    xgb_class_preds = pd.Series(xgb_class_preds_num).map(REV_CLASS_MAPPING).values

    xgb_acc = float(accuracy_score(y_class_test, xgb_class_preds))
    xgb_prec = float(precision_score(y_class_test, xgb_class_preds, average="weighted", zero_division=0))
    xgb_rec = float(recall_score(y_class_test, xgb_class_preds, average="weighted", zero_division=0))
    xgb_f1 = float(f1_score(y_class_test, xgb_class_preds, average="weighted", zero_division=0))
    xgb_cm = confusion_matrix(y_class_test, xgb_class_preds, labels=classes_list).tolist()

    # ========================================================
    # 3. RANDOM FOREST REGRESSOR
    # ========================================================
    rf_reg_pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_reg_pipeline.fit(X_train, y_reg_train)
    rf_reg_preds = rf_reg_pipeline.predict(X_test)

    rf_mae = float(mean_absolute_error(y_reg_test, rf_reg_preds))
    rf_mse = float(mean_squared_error(y_reg_test, rf_reg_preds))
    rf_rmse = float(np.sqrt(rf_mse))
    rf_r2 = float(r2_score(y_reg_test, rf_reg_preds))

    # ========================================================
    # 4. XGBOOST REGRESSOR
    # ========================================================
    xgb_reg_pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("regressor", xgb.XGBRegressor(n_estimators=100, random_state=42))
    ])
    xgb_reg_pipeline.fit(X_train, y_reg_train)
    xgb_reg_preds = xgb_reg_pipeline.predict(X_test)

    xgb_mae = float(mean_absolute_error(y_reg_test, xgb_reg_preds))
    xgb_mse = float(mean_squared_error(y_reg_test, xgb_reg_preds))
    xgb_rmse = float(np.sqrt(xgb_mse))
    xgb_r2 = float(r2_score(y_reg_test, xgb_reg_preds))

    # Optionally persist the primary production models
    if save_models:
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_clf_pipeline, "models/engagement_classifier.pkl")
        joblib.dump(rf_reg_pipeline, "models/academic_regressor.pkl")
        # Also maintain root fallback
        joblib.dump(rf_clf_pipeline, "engagement_classifier.pkl")
        joblib.dump(rf_reg_pipeline, "academic_regressor.pkl")

    results = {
        "dataset_summary": {
            "total_samples": len(df_clean),
            "train_samples": n_train,
            "test_samples": n_test,
            "classes": classes_list
        },
        "classification": {
            "random_forest": {
                "accuracy": rf_acc,
                "precision": rf_prec,
                "recall": rf_rec,
                "f1_score": rf_f1,
                "confusion_matrix": rf_cm,
                "report": rf_report
            },
            "xgboost": {
                "accuracy": xgb_acc,
                "precision": xgb_prec,
                "recall": xgb_rec,
                "f1_score": xgb_f1,
                "confusion_matrix": xgb_cm
            }
        },
        "regression": {
            "random_forest": {
                "mae": rf_mae,
                "mse": rf_mse,
                "rmse": rf_rmse,
                "r2": rf_r2
            },
            "xgboost": {
                "mae": xgb_mae,
                "mse": xgb_mse,
                "rmse": xgb_rmse,
                "r2": xgb_r2
            }
        },
        "feature_importances": feat_imp_dict,
        "models": {
            "rf_classifier": rf_clf_pipeline,
            "rf_regressor": rf_reg_pipeline,
            "xgb_classifier": xgb_clf_pipeline,
            "xgb_regressor": xgb_reg_pipeline
        }
    }

    return results
