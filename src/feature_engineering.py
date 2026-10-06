"""
Feature Engineering and Behavioral Learning Intelligence Module
"""

import pandas as pd
import numpy as np


POSITIVE_EMOTIONS = {"Happy", "Surprised"}
NEUTRAL_EMOTIONS = {"Neutral"}
NEGATIVE_EMOTIONS = {"Sad", "Scared", "Angry", "Disgust"}


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes behavioral indicators for student analytics and faculty insights:
    - Emotion_Valence: Positive, Neutral, or Negative
    - Activity_Intensity: Low, Moderate, High based on Weekly_Clicks
    - Clicks_Per_Engagement: Ratio of clicks to engagement score (interaction intensity)
    - Delivery_Click_Percentile: Relative click activity compared to delivery method cohort
    """
    df_feat = df.copy()

    # 1. Emotion Valence
    if "Primary_Emotion" in df_feat.columns:
        def categorize_emotion(e):
            if e in POSITIVE_EMOTIONS:
                return "Positive"
            elif e in NEGATIVE_EMOTIONS:
                return "Negative"
            return "Neutral"
        df_feat["Emotion_Valence"] = df_feat["Primary_Emotion"].apply(categorize_emotion)

    # 2. Activity Intensity
    if "Weekly_Clicks" in df_feat.columns:
        df_feat["Activity_Intensity"] = pd.cut(
            df_feat["Weekly_Clicks"],
            bins=[-1, 60, 140, 99999],
            labels=["Low Activity", "Moderate Activity", "High Activity"]
        ).astype(str)

    # 3. Clicks to Engagement Ratio
    if "Weekly_Clicks" in df_feat.columns and "Engagement_Score" in df_feat.columns:
        safe_eng = df_feat["Engagement_Score"].replace(0, 0.001)
        df_feat["Clicks_Per_Engagement"] = (df_feat["Weekly_Clicks"] / safe_eng).round(1)

    # 4. Cohort relative clicks
    if "Delivery_Method" in df_feat.columns and "Weekly_Clicks" in df_feat.columns:
        df_feat["Cohort_Click_Avg"] = df_feat.groupby("Delivery_Method")["Weekly_Clicks"].transform("mean").round(1)
        df_feat["Clicks_Vs_Cohort"] = (df_feat["Weekly_Clicks"] - df_feat["Cohort_Click_Avg"]).round(1)

    return df_feat
