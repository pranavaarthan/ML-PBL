import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)

def load_or_generate_dataset(csv_filepath="learner_engagement_dataset_3000.csv", num_samples=3000):
    """
    Loads dataset from CSV if available; otherwise generates a synthetic fallback
    matching the schema and statistical distribution.
    """
    if os.path.exists(csv_filepath):
        print(f"[INFO] Loading dataset from '{csv_filepath}'...")
        df = pd.read_csv(csv_filepath)
    else:
        print(f"[INFO] '{csv_filepath}' not found. Generating synthetic dataset ({num_samples} samples)...")
        np.random.seed(42)
        student_ids = [f"STU_{i:04d}" for i in range(1, num_samples + 1)]
        delivery_methods = np.random.choice(["Online", "Hybrid", "In-Person"], size=num_samples, p=[0.4, 0.3, 0.3])
        
        emotions_list = ["Neutral", "Happy", "Surprised", "Sad", "Scared", "Angry", "Disgust"]
        emotion_probs = [0.35, 0.25, 0.10, 0.12, 0.08, 0.06, 0.04]
        primary_emotions = np.random.choice(emotions_list, size=num_samples, p=emotion_probs)

        weekly_clicks = []
        engagement_scores = []
        engagement_levels = []
        academic_scores = []

        for emotion in primary_emotions:
            if emotion in ["Neutral", "Happy", "Surprised"]:
                clicks = np.random.randint(80, 250)
                score = np.random.uniform(0.10, 0.22)
            else:
                clicks = np.random.randint(5, 90)
                score = np.random.uniform(0.01, 0.099)

            weekly_clicks.append(clicks)
            engagement_scores.append(round(score, 3))

            if score > 0.14:
                engagement_levels.append("Highly Engaged")
                ac_score = np.random.uniform(88.0, 100.0)
            elif 0.10 <= score <= 0.14:
                engagement_levels.append("Engaged")
                ac_score = np.random.uniform(75.0, 92.0)
            else:
                engagement_levels.append("Disengaged")
                ac_score = np.random.uniform(40.0, 74.0)

            academic_scores.append(round(ac_score, 1))

        df = pd.DataFrame({
            "Student_ID": student_ids,
            "Delivery_Method": delivery_methods,
            "Weekly_Clicks": weekly_clicks,
            "Primary_Emotion": primary_emotions,
            "Engagement_Score": engagement_scores,
            "Engagement_Level": engagement_levels,
            "Academic_Success_Score": academic_scores
        })
        df.to_csv(csv_filepath, index=False)
        print(f"[INFO] Synthetic dataset created and saved to '{csv_filepath}'.")

    return df

def prepare_data(df):
    """
    Separates predictors and target variables.
    Drops Student_ID as it is a unique identifier.
    """
    # Features (Inputs)
    X = df[["Delivery_Method", "Weekly_Clicks", "Primary_Emotion", "Engagement_Score"]]
    
    # Target 1: Categorical Classification (Engagement_Level)
    y_class = df["Engagement_Level"]
    
    # Target 2: Continuous Regression (Academic_Success_Score)
    y_reg = df["Academic_Success_Score"]

    return X, y_class, y_reg

def get_preprocessor():
    """
    Builds a scikit-learn ColumnTransformer for preprocessing:
    - OneHotEncoder for categorical columns (Delivery_Method, Primary_Emotion)
    - StandardScaler for numerical columns (Weekly_Clicks, Engagement_Score)
    """
    categorical_cols = ["Delivery_Method", "Primary_Emotion"]
    numerical_cols = ["Weekly_Clicks", "Engagement_Score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor

def train_engagement_classifier(X_train, y_train, preprocessor):
    """
    Trains Random Forest Classifier to predict Engagement_Level.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_academic_regressor(X_train, y_train, preprocessor):
    """
    Trains Random Forest Regressor to predict Academic_Success_Score.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_models(clf_model, reg_model, X_test, y_class_test, y_reg_test):
    """
    Evaluates both classification and regression model performances.
    """
    print("\n" + "="*50)
    print(" 1. ENGAGEMENT LEVEL CLASSIFICATION EVALUATION ")
    print("="*50)
    class_preds = clf_model.predict(X_test)
    acc = accuracy_score(y_class_test, class_preds)
    print(f"Classifier Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_class_test, class_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_class_test, class_preds))

    print("\n" + "="*50)
    print(" 2. ACADEMIC SUCCESS REGRESSION EVALUATION ")
    print("="*50)
    reg_preds = reg_model.predict(X_test)
    mse = mean_squared_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, reg_preds)
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared Score (R²): {r2:.3f}")

def predict_sample(clf_model, reg_model, sample_dict):
    """
    Demonstrates inference on a new student log entry.
    """
    sample_df = pd.DataFrame([sample_dict])
    pred_level = clf_model.predict(sample_df)[0]
    pred_score = reg_model.predict(sample_df)[0]
    return pred_level, pred_score

if __name__ == "__main__":
    # 1. Load Data
    df = load_or_generate_dataset()

    # 2. Prepare Features & Targets
    X, y_class, y_reg = prepare_data(df)

    # 3. Train/Test Split (80/20 ratio)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    # 4. Get Preprocessor
    preprocessor = get_preprocessor()

    # 5. Train Models
    print("\n[INFO] Training Random Forest Classifier for Engagement_Level...")
    clf_model = train_engagement_classifier(X_train, y_class_train, preprocessor)

    print("[INFO] Training Random Forest Regressor for Academic_Success_Score...")
    reg_model = train_academic_regressor(X_train, y_reg_train, preprocessor)

    # 6. Evaluate Models
    evaluate_models(clf_model, reg_model, X_test, y_class_test, y_reg_test)

    # 7. Save Models
    joblib.dump(clf_model, "engagement_classifier.pkl")
    joblib.dump(reg_model, "academic_regressor.pkl")
    print("\n[INFO] Models saved successfully as 'engagement_classifier.pkl' and 'academic_regressor.pkl'.")

    # 8. Test Sample Inference
    test_student = {
        "Delivery_Method": "Online",
        "Weekly_Clicks": 145,
        "Primary_Emotion": "Neutral",
        "Engagement_Score": 0.152
    }
    
    predicted_level, predicted_academic = predict_sample(clf_model, reg_model, test_student)
    print("\n" + "="*50)
    print(" SAMPLE INFERENCE TEST ")
    print("="*50)
    print(f"Input Features: {test_student}")
    print(f"Predicted Engagement Level: {predicted_level}")
    print(f"Predicted Academic Score:   {predicted_academic:.1f}%")