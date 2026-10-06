"""
Transparent Rule & Model-Based At-Risk Student Detection Module
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


NEGATIVE_EMOTIONS = {"Sad", "Scared", "Angry", "Disgust"}


def evaluate_student_risk(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates risk level and generates reasons and interventions for an individual student.
    
    Signals evaluated:
    - Engagement level (Disengaged, Engaged, Highly Engaged)
    - Engagement score (< 0.10, < 0.06)
    - Weekly clicks (< 70, < 35)
    - Academic score (predicted or actual < 70, < 55)
    - Negative emotional indicators (contextual factor, never flags risk on its own)
    """
    risk_points = 0
    reasons: List[str] = []

    # 1. Engagement Level
    eng_level = row.get("Predicted_Engagement_Level") or row.get("Engagement_Level")
    if eng_level == "Disengaged":
        risk_points += 3
        reasons.append("Engagement status is classified as Disengaged.")
    elif eng_level == "Engaged":
        risk_points += 0

    # 2. Engagement Score
    eng_score = row.get("Engagement_Score")
    if eng_score is not None:
        try:
            score_val = float(eng_score)
            if score_val < 0.06:
                risk_points += 3
                reasons.append(f"Critically low engagement score: {score_val:.3f} (alert threshold < 0.060).")
            elif score_val < 0.10:
                risk_points += 2
                reasons.append(f"Low engagement score: {score_val:.3f} (below standard threshold of 0.100).")
        except (ValueError, TypeError):
            pass

    # 3. Weekly Clicks (Activity Volume)
    clicks = row.get("Weekly_Clicks")
    if clicks is not None:
        try:
            clicks_val = float(clicks)
            if clicks_val < 30:
                risk_points += 3
                reasons.append(f"Severely depressed LMS interaction: {int(clicks_val)} weekly clicks (critical < 30).")
            elif clicks_val < 65:
                risk_points += 2
                reasons.append(f"Low weekly interaction: {int(clicks_val)} weekly clicks (cohort benchmark >= 65).")
        except (ValueError, TypeError):
            pass

    # 4. Academic Score (Predicted or Actual)
    acad_score = row.get("Predicted_Academic_Score")
    if acad_score is None:
        acad_score = row.get("Academic_Success_Score")

    if acad_score is not None:
        try:
            acad_val = float(acad_score)
            if acad_val < 55.0:
                risk_points += 3
                reasons.append(f"Critically vulnerable academic trajectory: {acad_val:.1f}% (passing threshold >= 55.0%).")
            elif acad_val < 70.0:
                risk_points += 2
                reasons.append(f"Suboptimal academic performance: {acad_val:.1f}% (below proficiency standard of 70.0%).")
        except (ValueError, TypeError):
            pass

    # 5. Emotional Context (Emotional indicator only contributes if other risk signals exist)
    emotion = row.get("Primary_Emotion")
    if emotion in NEGATIVE_EMOTIONS and risk_points >= 2:
        risk_points += 1
        reasons.append(f"Contextual affective stress: Frequent '{emotion}' emotion logged alongside reduced participation.")

    # Determine Risk Level Category
    if risk_points >= 6 or (eng_level == "Disengaged" and acad_score is not None and float(acad_score) < 65.0):
        risk_level = "High"
        intervention = (
            "Immediate Faculty Follow-Up Required: Schedule an academic advisory session, "
            "review course access barriers, and initiate a personalized remediation plan."
        )
    elif risk_points >= 3:
        risk_level = "Medium"
        intervention = (
            "Targeted Academic Monitoring: Dispatch an engagement check-in prompt, verify "
            "weekly assignment completion, and offer peer tutoring resources."
        )
    else:
        risk_level = "Low"
        intervention = (
            "Student On Track: Consistent participation observed. Maintain current pedagogical pacing."
        )

    if not reasons:
        reasons.append("All behavioral engagement metrics and academic performance are within healthy ranges.")

    return {
        "risk_level": risk_level,
        "risk_score": risk_points,
        "reasons": reasons,
        "reasons_str": " • " + " • ".join(reasons),
        "intervention": intervention
    }


def assess_batch_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches an entire dataframe with risk assessments.
    Adds:
    - Risk_Level: ['Low', 'Medium', 'High']
    - Risk_Score: integer risk points
    - Risk_Reasons: bullet list string of reasons
    - Recommended_Intervention: actionable faculty guidance string
    """
    df_assessed = df.copy()

    risk_levels = []
    risk_scores = []
    risk_reasons_list = []
    interventions = []

    for _, row in df_assessed.iterrows():
        eval_res = evaluate_student_risk(row.to_dict())
        risk_levels.append(eval_res["risk_level"])
        risk_scores.append(eval_res["risk_score"])
        risk_reasons_list.append(eval_res["reasons_str"])
        interventions.append(eval_res["intervention"])

    df_assessed["Risk_Level"] = risk_levels
    df_assessed["Risk_Score"] = risk_scores
    df_assessed["Risk_Reasons"] = risk_reasons_list
    df_assessed["Recommended_Intervention"] = interventions

    return df_assessed
