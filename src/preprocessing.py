"""
Data Ingestion, Validation, and Preprocessing Module
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical schema definitions
FEATURE_NUMERICAL = ["Weekly_Clicks", "Engagement_Score"]
FEATURE_CATEGORICAL = ["Delivery_Method", "Primary_Emotion"]
ALL_FEATURES = FEATURE_NUMERICAL + FEATURE_CATEGORICAL

TARGET_CLASSIFICATION = "Engagement_Level"
TARGET_REGRESSION = "Academic_Success_Score"
ID_COLUMN = "Student_ID"

VALID_DELIVERY_METHODS = ["Online", "Hybrid", "In-Person"]
VALID_EMOTIONS = ["Neutral", "Happy", "Surprised", "Sad", "Scared", "Angry", "Disgust"]
VALID_ENGAGEMENT_LEVELS = ["Disengaged", "Engaged", "Highly Engaged"]

DEFAULT_CSV_PATHS = [
    "data/learner_engagement_dataset_3000.csv",
    "learner_engagement_dataset_3000.csv"
]


def find_default_dataset_path() -> Optional[str]:
    """Finds the default CSV dataset in known locations."""
    for path in DEFAULT_CSV_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_dataset(csv_filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Dynamically loads the dataset from a specified filepath or known default locations.
    Raises FileNotFoundError if no dataset is found.
    """
    if csv_filepath and os.path.exists(csv_filepath):
        target_path = csv_filepath
    else:
        target_path = find_default_dataset_path()

    if not target_path:
        raise FileNotFoundError(
            "Dataset file not found. Looked in: " + ", ".join(DEFAULT_CSV_PATHS)
        )

    df = pd.read_csv(target_path)
    return df


def validate_dataset(df: pd.DataFrame, require_targets: bool = False) -> Dict[str, Any]:
    """
    Validates the structure, columns, data types, and values of a student dataset.
    Returns a comprehensive validation dictionary.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return {
            "is_valid": False,
            "errors": ["Uploaded dataset is empty."],
            "warnings": [],
            "row_count": 0,
            "duplicate_count": 0,
            "missing_values": {}
        }

    # 1. Required column check
    required_cols = list(ALL_FEATURES)
    if require_targets:
        required_cols.extend([TARGET_CLASSIFICATION, TARGET_REGRESSION])

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # 2. Row count and duplicate check
    row_count = len(df)
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate rows in dataset.")

    # 3. Missing values check
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    if total_missing > 0:
        warnings.append(f"Found {total_missing} missing values across columns.")

    # 4. Numerical range checks
    if "Weekly_Clicks" in df.columns:
        if not pd.to_numeric(df["Weekly_Clicks"], errors="coerce").notnull().all():
            errors.append("'Weekly_Clicks' contains non-numeric values.")
        else:
            neg_clicks = (df["Weekly_Clicks"] < 0).sum()
            if neg_clicks > 0:
                warnings.append(f"Found {neg_clicks} rows with negative Weekly_Clicks; these should be >= 0.")

    if "Engagement_Score" in df.columns:
        if not pd.to_numeric(df["Engagement_Score"], errors="coerce").notnull().all():
            errors.append("'Engagement_Score' contains non-numeric values.")
        else:
            invalid_eng = ((df["Engagement_Score"] < 0.0) | (df["Engagement_Score"] > 1.0)).sum()
            if invalid_eng > 0:
                warnings.append(f"Found {invalid_eng} rows with Engagement_Score outside standard range [0.0, 1.0].")

    if TARGET_REGRESSION in df.columns:
        if pd.to_numeric(df[TARGET_REGRESSION], errors="coerce").isnull().any():
            warnings.append(f"'{TARGET_REGRESSION}' contains non-numeric entries.")

    # 5. Categorical check
    if "Delivery_Method" in df.columns:
        unexpected_methods = set(df["Delivery_Method"].dropna().unique()) - set(VALID_DELIVERY_METHODS)
        if unexpected_methods:
            warnings.append(f"Found unexpected delivery methods: {unexpected_methods}. Standard: {VALID_DELIVERY_METHODS}")

    if "Primary_Emotion" in df.columns:
        unexpected_emotions = set(df["Primary_Emotion"].dropna().unique()) - set(VALID_EMOTIONS)
        if unexpected_emotions:
            warnings.append(f"Found unexpected primary emotions: {unexpected_emotions}. Standard: {VALID_EMOTIONS}")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": row_count,
        "duplicate_count": duplicate_count,
        "missing_values": missing_dict
    }


def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw student dataset:
    - Removes duplicates
    - Imputes missing numeric values with medians
    - Imputes missing categorical values with modes
    - Clamps numerical ranges to valid boundaries
    """
    df_clean = df.copy()

    # Drop pure duplicates
    df_clean = df_clean.drop_duplicates()

    # Handle Student_ID
    if ID_COLUMN not in df_clean.columns:
        df_clean[ID_COLUMN] = [f"STU_{i:04d}" for i in range(1, len(df_clean) + 1)]

    # Clean numerical columns
    if "Weekly_Clicks" in df_clean.columns:
        df_clean["Weekly_Clicks"] = pd.to_numeric(df_clean["Weekly_Clicks"], errors="coerce")
        median_clicks = df_clean["Weekly_Clicks"].median() if not df_clean["Weekly_Clicks"].dropna().empty else 100
        df_clean["Weekly_Clicks"] = df_clean["Weekly_Clicks"].fillna(median_clicks).clip(lower=0)

    if "Engagement_Score" in df_clean.columns:
        df_clean["Engagement_Score"] = pd.to_numeric(df_clean["Engagement_Score"], errors="coerce")
        median_eng = df_clean["Engagement_Score"].median() if not df_clean["Engagement_Score"].dropna().empty else 0.12
        df_clean["Engagement_Score"] = df_clean["Engagement_Score"].fillna(median_eng).clip(lower=0.0, upper=1.0)

    # Clean categorical columns
    if "Delivery_Method" in df_clean.columns:
        mode_method = df_clean["Delivery_Method"].mode()[0] if not df_clean["Delivery_Method"].dropna().empty else "Online"
        df_clean["Delivery_Method"] = df_clean["Delivery_Method"].fillna(mode_method).astype(str)

    if "Primary_Emotion" in df_clean.columns:
        mode_emotion = df_clean["Primary_Emotion"].mode()[0] if not df_clean["Primary_Emotion"].dropna().empty else "Neutral"
        df_clean["Primary_Emotion"] = df_clean["Primary_Emotion"].fillna(mode_emotion).astype(str)

    if TARGET_REGRESSION in df_clean.columns:
        df_clean[TARGET_REGRESSION] = pd.to_numeric(df_clean[TARGET_REGRESSION], errors="coerce")
        if df_clean[TARGET_REGRESSION].notnull().any():
            median_score = df_clean[TARGET_REGRESSION].median()
            df_clean[TARGET_REGRESSION] = df_clean[TARGET_REGRESSION].fillna(median_score).clip(lower=0.0, upper=100.0)

    return df_clean


def get_preprocessor() -> ColumnTransformer:
    """
    Builds a standard scikit-learn ColumnTransformer:
    - StandardScaler for numerical columns (Weekly_Clicks, Engagement_Score)
    - OneHotEncoder for categorical columns (Delivery_Method, Primary_Emotion)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERICAL),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CATEGORICAL),
        ]
    )
    return preprocessor


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Separates predictors X and optional targets y_class, y_reg.
    """
    # Features required for model prediction
    X = df[ALL_FEATURES].copy()

    y_class = df[TARGET_CLASSIFICATION].copy() if TARGET_CLASSIFICATION in df.columns else None
    y_reg = df[TARGET_REGRESSION].copy() if TARGET_REGRESSION in df.columns else None

    return X, y_class, y_reg
