# ==========================================
# STUDENT STRESS LEVEL - MODEL TRAINING + EDA
# ==========================================

# ------------------------------------------
# STEP 0: Import Required Libraries
# ------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------
data_path = "../dataset/Student Stress Factors.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(df.head())

# ------------------------------------------
# STEP 2: Basic Data Understanding (EDA)
# ------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------
# STEP 3: Check Missing Values
# ------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# STEP 4: Target Variable Distribution (EDA)
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="stress_level", data=df)
plt.title("Stress Level Distribution")
plt.xlabel("Stress Level (1–5)")
plt.ylabel("Number of Students")
plt.show()

# ------------------------------------------
# STEP 5: Feature Distributions (EDA)
# ------------------------------------------
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# ------------------------------------------
# STEP 6: Correlation Heatmap (EDA)
# ------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# STEP 7: Feature vs Target Analysis (EDA)
# ------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="stress_level", y="study_load", data=df)
plt.title("Study Load vs Stress Level")
plt.show()

# ------------------------------------------
# STEP 8: Split Features & Target
# ------------------------------------------
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

# ------------------------------------------
# STEP 9: Feature Scaling
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------
# STEP 10: Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ------------------------------------------
# STEP 11: Train Logistic Regression Model
# ------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed successfully")

# ------------------------------------------
# STEP 12: Model Evaluation
# ------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------
# STEP 13: Save Model & Scaler
# ------------------------------------------
pickle.dump(model, open("../model/stress_model.pkl", "wb"))
pickle.dump(scaler, open("../model/scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")
