import os
import joblib
import pandas as pd

print("Checking files...")
print("CSV exists:", os.path.exists("data/learner_engagement_dataset_3000.csv"))
print("CLF exists:", os.path.exists("models/engagement_classifier.pkl"))
print("REG exists:", os.path.exists("models/academic_regressor.pkl"))

df = pd.read_csv("data/learner_engagement_dataset_3000.csv")
print(f"Dataset shape: {df.shape}")
print("Columns:", list(df.columns))
print("Missing values:\n", df.isnull().sum())
print("Engagement levels:\n", df["Engagement_Level"].value_counts())
print("Primary emotions:\n", df["Primary_Emotion"].value_counts())
print("Delivery methods:\n", df["Delivery_Method"].value_counts())

clf = joblib.load("models/engagement_classifier.pkl")
reg = joblib.load("models/academic_regressor.pkl")

print("\nClassifier Pipeline steps:", clf.named_steps if hasattr(clf, "named_steps") else type(clf))
print("Regressor Pipeline steps:", reg.named_steps if hasattr(reg, "named_steps") else type(reg))

sample = df.iloc[:3][["Delivery_Method", "Weekly_Clicks", "Primary_Emotion", "Engagement_Score"]]
print("\nSample test prediction:")
print("True Engagement:", list(df.iloc[:3]["Engagement_Level"]))
print("Pred Engagement:", list(clf.predict(sample)))
print("True Academic:", list(df.iloc[:3]["Academic_Success_Score"]))
print("Pred Academic:", list(reg.predict(sample)))
