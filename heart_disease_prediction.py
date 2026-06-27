
# ==============================================
# Heart Disease Prediction using Machine Learning
# Part 1: Import Libraries, Load Data,
# Preprocessing and Exploratory Data Analysis
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create required folders
os.makedirs("results", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/heart.csv")

print("="*60)
print("First Five Rows")
print("="*60)
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nShape After Removing Duplicates")
print(df.shape)

# --------------------------------------
# Target Distribution
# --------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("results/target_distribution.png")
plt.show()

# --------------------------------------
# Age Distribution
# --------------------------------------

plt.figure(figsize=(7,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("results/age_distribution.png")
plt.show()

# --------------------------------------
# Gender vs Target
# --------------------------------------

plt.figure(figsize=(7,5))
sns.countplot(x="sex", hue="target", data=df)
plt.title("Gender vs Heart Disease")
plt.savefig("results/gender_vs_target.png")
plt.show()

# --------------------------------------
# Chest Pain Type
# --------------------------------------

plt.figure(figsize=(7,5))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Chest Pain Type")
plt.savefig("results/chest_pain.png")
plt.show()

# --------------------------------------
# Cholesterol Distribution
# --------------------------------------

plt.figure(figsize=(7,5))
sns.boxplot(x="target", y="chol", data=df)
plt.title("Cholesterol Distribution")
plt.savefig("results/cholesterol_boxplot.png")
plt.show()

# --------------------------------------
# Blood Pressure Distribution
# --------------------------------------

plt.figure(figsize=(7,5))
sns.boxplot(x="target", y="trestbps", data=df)
plt.title("Resting Blood Pressure")
plt.savefig("results/blood_pressure.png")
plt.show()

# --------------------------------------
# Maximum Heart Rate
# --------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="age",
    y="thalach",
    hue="target",
    data=df
)
plt.title("Age vs Maximum Heart Rate")
plt.savefig("results/heart_rate.png")
plt.show()

# --------------------------------------
# Correlation Matrix
# --------------------------------------

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidth=0.5
)

plt.title("Correlation Matrix")
plt.savefig("results/correlation_matrix.png")
plt.show()

# --------------------------------------
# Pair Plot
# --------------------------------------

sns.pairplot(
    df,
    hue="target"
)

plt.savefig("results/pairplot.png")
plt.show()

# --------------------------------------
# Feature and Target
# --------------------------------------

X = df.drop("target", axis=1)

y = df["target"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# --------------------------------------
# Feature Scaling
# --------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# --------------------------------------
# Train Test Split
# --------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

print("\nPart 1 Completed Successfully")

# ==============================================
# PART 2 : MODEL TRAINING AND EVALUATION
# ==============================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib

# Dictionary to store models

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Support Vector Machine": SVC(),

    "K-Nearest Neighbors": KNeighborsClassifier()

}

# Store Results

results = []

best_accuracy = 0
best_model = None
best_name = ""

# ======================================
# Train Every Model
# ======================================

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    # Train Model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report\n")

    print(classification_report(y_test, y_pred))

    # Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(name + " Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    filename = name.lower().replace(" ", "_") + "_cm.png"

    plt.savefig("results/" + filename)

    plt.show()

    # Save Results

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1

    ])

    # Best Model

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name

# ======================================
# Result Table
# ======================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

print("\n")
print(results_df)

# Save CSV

results_df.to_csv(

    "results/evaluation_metrics.csv",

    index=False

)

# ======================================
# Accuracy Comparison Graph
# ======================================

plt.figure(figsize=(10,6))

sns.barplot(

    x="Model",

    y="Accuracy",

    data=results_df

)

plt.xticks(rotation=20)

plt.title("Model Accuracy Comparison")

plt.savefig("results/model_comparison.png")

plt.show()

# ======================================
# Feature Importance
# ======================================

if best_name == "Random Forest":

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    print("\nFeature Importance\n")

    print(importance)

    plt.figure(figsize=(10,6))

    sns.barplot(

        x="Importance",

        y="Feature",

        data=importance

    )

    plt.title("Feature Importance")

    plt.savefig("results/feature_importance.png")

    plt.show()

# ======================================
# Save Best Model
# ======================================

joblib.dump(

    best_model,

    "model/heart_disease_model.pkl"

)

print("\n" + "="*60)

print("Best Model :", best_name)

print("Accuracy   :", best_accuracy)

print("Model Saved Successfully")

print("="*60)

# ==============================================
# PART 3 : PREDICT NEW PATIENT
# ==============================================

print("\n")
print("="*60)
print("PREDICT HEART DISEASE FOR NEW PATIENT")
print("="*60)

# Example Patient Data
# Change these values if required

new_patient = [[
    52,   # age
    1,    # sex
    0,    # cp
    125,  # trestbps
    212,  # chol
    0,    # fbs
    1,    # restecg
    168,  # thalach
    0,    # exang
    1.0,  # oldpeak
    2,    # slope
    0,    # ca
    2     # thal
]]

# Scale the input
new_patient_scaled = scaler.transform(new_patient)

# Predict
prediction = best_model.predict(new_patient_scaled)

if prediction[0] == 1:
    print("\nPrediction : Heart Disease Detected")
else:
    print("\nPrediction : No Heart Disease")

print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)