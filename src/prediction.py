"""
Prediction and Inference Pipeline Module
"""

import os
from typing import Tuple, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import ALL_FEATURES, clean_and_impute_data

MODEL_DIR = "models"
CLASSIFIER_FILENAMES = ["models/engagement_classifier.pkl", "engagement_classifier.pkl"]
REGRESSOR_FILENAMES = ["models/academic_regressor.pkl", "academic_regressor.pkl"]


def find_model_file(candidate_paths: list) -> Optional[str]:
    """Finds the first existing file among candidate paths."""
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_models() -> Tuple[Any, Any]:
    """
    Safely loads the engagement classifier and academic regressor.
    Returns:
        (classifier_model, regressor_model)
    Raises:
        FileNotFoundError: If either model file cannot be located.
        Exception: If model deserialization or validation fails.
    """
    clf_path = find_model_file(CLASSIFIER_FILENAMES)
    reg_path = find_model_file(REGRESSOR_FILENAMES)

    if not clf_path:
        raise FileNotFoundError(
            f"Engagement Classifier not found. Looked in: {CLASSIFIER_FILENAMES}"
        )
    if not reg_path:
        raise FileNotFoundError(
            f"Academic Regressor not found. Looked in: {REGRESSOR_FILENAMES}"
        )

    try:
        clf_model = joblib.load(clf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Classifier from '{clf_path}': {str(e)}")

    try:
        reg_model = joblib.load(reg_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Regressor from '{reg_path}': {str(e)}")

    # Validate that models possess predict method
    if not hasattr(clf_model, "predict"):
        raise ValueError(f"Classifier loaded from '{clf_path}' does not have a predict method.")
    if not hasattr(reg_model, "predict"):
        raise ValueError(f"Regressor loaded from '{reg_path}' does not have a predict method.")

    return clf_model, reg_model


def predict_batch(clf_model: Any, reg_model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates model predictions for a full student dataframe.
    Adds:
    - Predicted_Engagement_Level
    - Engagement_Confidence (probability percentage)
    - Predicted_Academic_Score
    - Prediction_Error (if actual score present)
    """
    df_clean = clean_and_impute_data(df)
    X = df_clean[ALL_FEATURES]

    # Predict engagement level
    preds_class = clf_model.predict(X)
    df_clean["Predicted_Engagement_Level"] = preds_class

    # Predict engagement probabilities if available
    if hasattr(clf_model, "predict_proba"):
        try:
            probas = clf_model.predict_proba(X)
            classes = list(clf_model.classes_)
            max_probas = np.max(probas, axis=1)
            df_clean["Engagement_Confidence"] = (max_probas * 100).round(1)
            for i, cls_name in enumerate(classes):
                df_clean[f"Prob_{cls_name.replace(' ', '_')}"] = (probas[:, i] * 100).round(1)
        except Exception:
            df_clean["Engagement_Confidence"] = 100.0
    else:
        df_clean["Engagement_Confidence"] = 100.0

    # Predict academic score
    preds_reg = reg_model.predict(X)
    df_clean["Predicted_Academic_Score"] = np.round(np.clip(preds_reg, 0.0, 100.0), 1)

    # Compute error if ground truth is present
    if "Academic_Success_Score" in df_clean.columns:
        df_clean["Prediction_Error"] = (
            df_clean["Predicted_Academic_Score"] - df_clean["Academic_Success_Score"]
        ).round(1)

    return df_clean


def predict_single(clf_model: Any, reg_model: Any, student_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs dual inference on a single student dictionary record.
    """
    df_single = pd.DataFrame([student_dict])
    df_result = predict_batch(clf_model, reg_model, df_single)
    row = df_result.iloc[0].to_dict()

    return {
        "predicted_engagement_level": row.get("Predicted_Engagement_Level"),
        "confidence": row.get("Engagement_Confidence", 100.0),
        "predicted_academic_score": row.get("Predicted_Academic_Score"),
        "raw_record": row
    }
