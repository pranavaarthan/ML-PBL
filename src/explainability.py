"""
Model Explainability Module using SHAP and Feature Attributions
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from src.preprocessing import ALL_FEATURES


def explain_student_prediction(
    pipeline: Any,
    student_df: pd.DataFrame,
    background_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes SHAP feature attributions for a single student prediction.
    Explains 'Why was this student classified this way?'.
    
    Returns:
        dict containing:
        - method: 'SHAP TreeExplainer' or 'Feature Importance'
        - predicted_class: str
        - top_factors: list of tuples (feature_name, contribution_weight, direction)
        - detailed_contributions: dict of feature -> shap value
        - figure: Plotly figure ready for rendering
    """
    if pipeline is None:
        raise ValueError("Pipeline cannot be None.")

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")

    if not preprocessor or not classifier:
        raise ValueError("Pipeline must contain 'preprocessor' and 'classifier'/'regressor' steps.")

    # Prepare single sample row
    sample_input = student_df[ALL_FEATURES].iloc[:1]
    predicted_class = pipeline.predict(sample_input)[0]

    # Preprocess the sample
    X_sample_trans = preprocessor.transform(sample_input)
    feature_names = preprocessor.get_feature_names_out()

    # Determine class index for multi-class classification
    classes = list(getattr(classifier, "classes_", []))
    class_idx = classes.index(predicted_class) if predicted_class in classes else 0

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_sample_trans)

        # Handle different SHAP output formats across shap versions
        if isinstance(shap_vals, list):
            # List of arrays [n_samples, n_features] per class
            sample_shap = shap_vals[class_idx][0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                # Shape: [n_samples, n_features, n_classes]
                sample_shap = shap_vals[0, :, class_idx]
            else:
                sample_shap = shap_vals[0]
        else:
            sample_shap = np.array(shap_vals)[0]

        method_used = "SHAP TreeExplainer"

    except Exception:
        # Fallback to feature importance weighting without fabricating values
        method_used = "Model Feature Importances"
        importances = getattr(classifier, "feature_importances_", np.ones(len(feature_names)))
        sample_shap = importances * (X_sample_trans[0] - np.mean(X_sample_trans[0]))

    # Map back to readable feature names and group by canonical parent feature
    raw_feature_impacts = {}
    for name, val in zip(feature_names, sample_shap):
        clean_name = name.replace("num__", "").replace("cat__", "")
        raw_feature_impacts[clean_name] = float(val)

    # Aggregate contributions by parent features:
    # Engagement_Score, Weekly_Clicks, Delivery_Method, Primary_Emotion
    parent_impacts = {
        "Engagement Score": 0.0,
        "Weekly Clicks": 0.0,
        "Delivery Method": 0.0,
        "Primary Emotion": 0.0
    }

    for feat_name, impact in raw_feature_impacts.items():
        if "Engagement_Score" in feat_name:
            parent_impacts["Engagement Score"] += impact
        elif "Weekly_Clicks" in feat_name:
            parent_impacts["Weekly Clicks"] += impact
        elif "Delivery_Method" in feat_name:
            parent_impacts["Delivery Method"] += impact
        elif "Primary_Emotion" in feat_name:
            parent_impacts["Primary Emotion"] += impact
        else:
            parent_impacts[feat_name] = impact

    # Rank factors by magnitude of absolute impact
    ranked_factors = sorted(
        parent_impacts.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    top_factors_formatted = []
    for rank, (name, val) in enumerate(ranked_factors, 1):
        direction = "Positive Influence" if val >= 0 else "Negative Influence"
        top_factors_formatted.append({
            "rank": rank,
            "feature": name,
            "impact": round(val, 4),
            "direction": direction
        })

    # Build interactive Plotly chart
    fig = create_shap_horizontal_bar(parent_impacts, predicted_class, method_used)

    return {
        "method": method_used,
        "predicted_class": predicted_class,
        "top_factors": top_factors_formatted,
        "parent_impacts": parent_impacts,
        "raw_feature_impacts": raw_feature_impacts,
        "figure": fig
    }


def create_shap_horizontal_bar(
    parent_impacts: Dict[str, float],
    predicted_class: str,
    method_used: str
) -> go.Figure:
    """
    Generates an explanatory horizontal bar chart of feature attributions.
    """
    features = list(parent_impacts.keys())
    values = list(parent_impacts.values())

    # Sort so highest impact is on top
    sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]))
    sorted_features = [p[0] for p in sorted_pairs]
    sorted_values = [p[1] for p in sorted_pairs]

    colors = [
        "#10b981" if v >= 0 else "#ef4444"
        for v in sorted_values
    ]

    fig = go.Figure(
        go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
            text=[f"{v:+.4f}" for v in sorted_values],
            textposition="auto"
        )
    )

    fig.update_layout(
        title=f"Feature Contributions for Class '{predicted_class}' ({method_used})",
        xaxis_title="Contribution Direction & Impact on Prediction",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=30),
        height=280
    )

    return fig
