"""
Unit and Integration Test Suite for Student Engagement Platform
"""

import unittest
import pandas as pd
import numpy as np

from src.preprocessing import (
    load_dataset,
    validate_dataset,
    clean_and_impute_data,
    prepare_data,
    ALL_FEATURES
)
from src.feature_engineering import add_behavioral_features
from src.prediction import load_models, predict_batch, predict_single
from src.risk_detection import evaluate_student_risk, assess_batch_risk
from src.explainability import explain_student_prediction
from src.model_training import train_and_evaluate_all
from utils.helpers import compute_what_changed


class TestStudentEngagementPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads dataset and production models once for testing."""
        cls.df = load_dataset()
        cls.clf_model, cls.reg_model = load_models()

    def test_01_dataset_loading(self):
        """Tests that default dataset exists and contains expected columns."""
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 1000)
        expected_cols = [
            "Student_ID", "Delivery_Method", "Weekly_Clicks",
            "Primary_Emotion", "Engagement_Score", "Engagement_Level",
            "Academic_Success_Score"
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_02_data_validation_valid(self):
        """Tests validation on valid student data."""
        report = validate_dataset(self.df.head(20))
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)

    def test_03_data_validation_invalid(self):
        """Tests validation failure when required features are missing."""
        invalid_df = pd.DataFrame([{"Student_ID": "STU_999", "Delivery_Method": "Online"}])
        report = validate_dataset(invalid_df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("Missing required columns" in err for err in report["errors"]))

    def test_04_data_cleaning_and_imputation(self):
        """Tests imputation of missing values and clipping of out-of-range values."""
        dirty_df = pd.DataFrame([
            {
                "Delivery_Method": None,
                "Weekly_Clicks": -50,  # invalid negative
                "Primary_Emotion": None,
                "Engagement_Score": 1.5,  # out of bounds (> 1.0)
            }
        ])
        cleaned = clean_and_impute_data(dirty_df)
        self.assertGreaterEqual(cleaned["Weekly_Clicks"].iloc[0], 0)
        self.assertLessEqual(cleaned["Engagement_Score"].iloc[0], 1.0)
        self.assertIsNotNone(cleaned["Delivery_Method"].iloc[0])
        self.assertIsNotNone(cleaned["Primary_Emotion"].iloc[0])

    def test_05_feature_engineering(self):
        """Tests behavioral indicators computation."""
        featured = add_behavioral_features(self.df.head(10))
        self.assertIn("Emotion_Valence", featured.columns)
        self.assertIn("Activity_Intensity", featured.columns)
        self.assertIn("Clicks_Per_Engagement", featured.columns)

    def test_06_model_loading_and_inference(self):
        """Tests model batch and single predictions."""
        sample_batch = self.df.head(10)
        scored = predict_batch(self.clf_model, self.reg_model, sample_batch)

        self.assertIn("Predicted_Engagement_Level", scored.columns)
        self.assertIn("Predicted_Academic_Score", scored.columns)
        self.assertIn("Engagement_Confidence", scored.columns)

        valid_levels = {"Disengaged", "Engaged", "Highly Engaged"}
        for level in scored["Predicted_Engagement_Level"]:
            self.assertIn(level, valid_levels)

        for score in scored["Predicted_Academic_Score"]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_07_risk_detection(self):
        """Tests transparent multi-signal risk detection logic."""
        # High risk profile
        high_risk_student = {
            "Predicted_Engagement_Level": "Disengaged",
            "Engagement_Score": 0.03,
            "Weekly_Clicks": 18,
            "Predicted_Academic_Score": 42.0,
            "Primary_Emotion": "Sad"
        }
        res_high = evaluate_student_risk(high_risk_student)
        self.assertEqual(res_high["risk_level"], "High")
        self.assertGreater(len(res_high["reasons"]), 1)
        self.assertIn("Immediate", res_high["intervention"])

        # Low risk profile
        low_risk_student = {
            "Predicted_Engagement_Level": "Highly Engaged",
            "Engagement_Score": 0.19,
            "Weekly_Clicks": 210,
            "Predicted_Academic_Score": 94.0,
            "Primary_Emotion": "Happy"
        }
        res_low = evaluate_student_risk(low_risk_student)
        self.assertEqual(res_low["risk_level"], "Low")

        # Test rule: Emotion alone does NOT trigger high risk
        emotion_only_student = {
            "Predicted_Engagement_Level": "Highly Engaged",
            "Engagement_Score": 0.18,
            "Weekly_Clicks": 190,
            "Predicted_Academic_Score": 92.0,
            "Primary_Emotion": "Sad"
        }
        res_emotion = evaluate_student_risk(emotion_only_student)
        self.assertNotEqual(res_emotion["risk_level"], "High")

    def test_08_explainability_shap(self):
        """Tests SHAP explainability generation for individual predictions."""
        sample_student = self.df.iloc[:1]
        explanation = explain_student_prediction(self.clf_model, sample_student)

        self.assertIn("method", explanation)
        self.assertIn("predicted_class", explanation)
        self.assertIn("top_factors", explanation)
        self.assertEqual(len(explanation["top_factors"]), 4)
        self.assertIsNotNone(explanation["figure"])

    def test_09_what_changed_engine(self):
        """Tests What Changed delta computation with and without history."""
        # Case 1: No history available
        res_no_hist = compute_what_changed(self.df, None)
        self.assertFalse(res_no_hist["has_history"])
        self.assertIn("Historical comparison data is not available.", res_no_hist["message"])

        # Case 2: History present
        prev_df = self.df.copy()
        prev_df["Weekly_Clicks"] += 10
        res_hist = compute_what_changed(self.df, prev_df)
        self.assertTrue(res_hist["has_history"])
        self.assertGreater(len(res_hist["bullet_points"]), 0)

    def test_10_model_training_and_comparison(self):
        """Tests training and comparative benchmarking of RF and XGBoost."""
        results = train_and_evaluate_all(self.df.head(200), save_models=False)

        self.assertIn("classification", results)
        self.assertIn("regression", results)

        rf_acc = results["classification"]["random_forest"]["accuracy"]
        xgb_acc = results["classification"]["xgboost"]["accuracy"]
        self.assertGreater(rf_acc, 0.70)
        self.assertGreater(xgb_acc, 0.70)

        rf_r2 = results["regression"]["random_forest"]["r2"]
        self.assertIsInstance(rf_r2, float)


if __name__ == "__main__":
    unittest.main()
