import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
if os.path.exists("cardio_train.csv"):
    df = pd.read_csv("cardio_train.csv", sep=";")
elif os.path.exists("cardio_train.csv.zip"):
    df = pd.read_csv("cardio_train.csv.zip", sep=";")
else:
    raise FileNotFoundError(
        "Place cardio_train.csv or cardio_train.csv.zip "
        "in the project folder."
    )

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Records ---")
print(df.duplicated().sum())

# 2. Data preprocessing
df = df.drop_duplicates()
df = df.drop(columns=["id"], errors="ignore")

# Age is stored in days in this dataset
if df["age"].max() > 150:
    df["age"] = (df["age"] / 365.25).round(1)

# Remove clearly invalid values
df = df[
    (df["height"].between(100, 250)) &
    (df["weight"].between(25, 300)) &
    (df["ap_hi"].between(60, 250)) &
    (df["ap_lo"].between(30, 200))
].copy()

# Remove records with inconsistent blood pressure
df = df[df["ap_hi"] >= df["ap_lo"]].copy()

print("\n--- Cleaned Dataset Shape ---")
print(df.shape)

print("\n--- Statistical Summary ---")
print(df.describe())

# 3. Exploratory Data Analysis (EDA)
os.makedirs("plots", exist_ok=True)

# Target distribution
sns.countplot(data=df, x="cardio")
plt.title("Heart Disease Class Distribution")
plt.xlabel("Cardio (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.show()
plt.close()

# Age distribution
sns.histplot(data=df, x="age", hue="cardio",
             bins=30, kde=True)
plt.title("Age Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig("plots/age_distribution.png")
plt.show()
plt.close()

# Gender vs heart disease
sns.countplot(data=df, x="gender", hue="cardio")
plt.title("Gender vs Heart Disease")
plt.tight_layout()
plt.savefig("plots/gender_disease.png")
plt.show()
plt.close()

# Cholesterol vs heart disease
sns.countplot(data=df, x="cholesterol", hue="cardio")
plt.title("Cholesterol vs Heart Disease")
plt.tight_layout()
plt.savefig("plots/cholesterol_disease.png")
plt.show()
plt.close()

# Blood pressure analysis
sns.boxplot(data=df, x="cardio", y="ap_hi")
plt.title("Systolic Blood Pressure vs Heart Disease")
plt.tight_layout()
plt.savefig("plots/blood_pressure.png")
plt.show()
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True, cmap="coolwarm",
            fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.show()
plt.close()

# 4. Prepare features and target
X = df.drop(columns=["cardio"])
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Train and compare 5 ML algorithms
models = {
    "SVM": make_pipeline(
        StandardScaler(), SVC()
    ),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=10
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42,
        n_jobs=-1, max_depth=15
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2)
    })
    trained_models[name] = model

    print("Accuracy:", round(accuracy * 100, 2), "%")

# 6. Compare model accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy (%)", ascending=False
)

print("\n--- Model Accuracy Comparison ---")
print(results_df.to_string(index=False))

results_df.to_csv("model_accuracy.csv", index=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df,
            x="Model", y="Accuracy (%)")
plt.title("Machine Learning Model Accuracy Comparison")
plt.xticks(rotation=25)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("plots/model_accuracy.png")
plt.show()
plt.close()

# 7. Evaluate the best model
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest model:", best_name)
print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

cm = confusion_matrix(y_test, best_predictions)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()
plt.close()

# 8. Save the best model
joblib.dump(best_model, "best_heart_disease_model.pkl")

print("\nProject completed successfully!")
print("Accuracy results saved to model_accuracy.csv")
print("Graphs saved in the plots folder.")
print("Best model saved as best_heart_disease_model.pkl")