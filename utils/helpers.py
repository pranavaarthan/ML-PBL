"""
UI, Theming, Formatting, and 'What Changed?' Analytics Helpers
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import streamlit as st


# Professional Academic Theme Styling
CUSTOM_CSS = """
<style>
    /* Global Typography & Background Accent */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* KPI Card Container */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .kpi-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-high-risk {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-medium-risk {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-low-risk {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }

    .badge-highly-engaged {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .badge-engaged {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .badge-disengaged {
        background-color: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    /* Notification / Alert Callout */
    .notice-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #334155;
    }

    /* Main container clean padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
</style>
"""


def apply_theme():
    """Injects custom academic theme styling into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: Any,
    subtext: str = "",
    accent_color: str = "#3b82f6"
):
    """Renders a styled KPI metric card."""
    html = f"""
    <div class="kpi-card" style="border-top: 3px solid {accent_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_risk_badge_html(risk_level: str) -> str:
    """Returns styled HTML badge for risk levels."""
    level = str(risk_level).lower()
    if level == "high":
        return f'<span class="badge badge-high-risk">⚠️ HIGH RISK</span>'
    elif level == "medium":
        return f'<span class="badge badge-medium-risk">⚡ MEDIUM RISK</span>'
    else:
        return f'<span class="badge badge-low-risk">✓ LOW RISK</span>'


def get_engagement_badge_html(eng_level: str) -> str:
    """Returns styled HTML badge for engagement levels."""
    level = str(eng_level).lower()
    if "highly" in level:
        return f'<span class="badge badge-highly-engaged">⭐ Highly Engaged</span>'
    elif "disengaged" in level:
        return f'<span class="badge badge-disengaged">🔻 Disengaged</span>'
    else:
        return f'<span class="badge badge-engaged">🔹 Engaged</span>'


def compute_what_changed(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes delta analytics between current period and historical period:
    - Change in Disengaged students count and percentage
    - Change in Average weekly clicks
    - Change in Average academic score
    - Additional students entering high-risk category
    
    If historical data is unavailable, cleanly returns 'Historical comparison data is not available.'
    """
    if previous_df is None or previous_df.empty:
        return {
            "has_history": False,
            "message": "Historical comparison data is not available.",
            "bullet_points": []
        }

    try:
        # 1. Disengaged change
        curr_dis = (current_df["Engagement_Level"] == "Disengaged").sum()
        prev_dis = (previous_df["Engagement_Level"] == "Disengaged").sum()
        dis_pct_change = (
            ((curr_dis - prev_dis) / prev_dis * 100) if prev_dis > 0 else 0.0
        )

        # 2. Weekly Clicks change
        curr_clicks = current_df["Weekly_Clicks"].mean()
        prev_clicks = previous_df["Weekly_Clicks"].mean()
        clicks_diff = curr_clicks - prev_clicks

        # 3. Academic Score change
        acad_col = "Academic_Success_Score"
        if acad_col not in current_df.columns:
            acad_col = "Predicted_Academic_Score"

        curr_acad = current_df[acad_col].mean()
        prev_acad = previous_df[acad_col].mean() if acad_col in previous_df.columns else curr_acad
        acad_diff = curr_acad - prev_acad

        # 4. High-risk students change
        curr_risk_high = (current_df.get("Risk_Level", pd.Series(["Low"])) == "High").sum()
        prev_risk_high = (previous_df.get("Risk_Level", pd.Series(["Low"])) == "High").sum()
        high_risk_diff = curr_risk_high - prev_risk_high

        # Build informative bullet points
        bullets: List[str] = []

        # Disengaged bullet
        if abs(dis_pct_change) > 0.1:
            dir_str = "increased" if dis_pct_change > 0 else "decreased"
            bullets.append(f"Disengaged students {dir_str} by {abs(dis_pct_change):.1f}%")
        else:
            bullets.append("Disengaged student proportions remained steady.")

        # Clicks bullet
        if abs(clicks_diff) >= 0.5:
            dir_str = "increased" if clicks_diff > 0 else "decreased"
            bullets.append(f"Average weekly clicks {dir_str} by {abs(clicks_diff):.1f}")
        else:
            bullets.append("Average weekly clicks remained consistent with previous period.")

        # Academic score bullet
        if abs(acad_diff) >= 0.1:
            dir_str = "increased" if acad_diff > 0 else "decreased"
            bullets.append(f"Average academic score {dir_str} by {abs(acad_diff):.1f} points")
        else:
            bullets.append("Average academic scores remained stable.")

        # High risk bullet
        if high_risk_diff > 0:
            bullets.append(f"{high_risk_diff} additional students entered the high-risk category")
        elif high_risk_diff < 0:
            bullets.append(f"{abs(high_risk_diff)} fewer students in the high-risk category (positive progress)")
        else:
            bullets.append("Total count of high-risk students remained unchanged.")

        return {
            "has_history": True,
            "message": "Comparison successfully computed against previous-period baseline.",
            "bullet_points": bullets,
            "metrics": {
                "disengaged_pct_change": dis_pct_change,
                "clicks_diff": clicks_diff,
                "acad_diff": acad_diff,
                "high_risk_diff": high_risk_diff
            }
        }

    except Exception as e:
        return {
            "has_history": False,
            "message": f"Historical comparison error: {str(e)}",
            "bullet_points": []
        }
