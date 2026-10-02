"""
SkillNexis - Week 2 Assignment
Logistic Regression on Titanic Dataset

Task:
Train a Logistic Regression model for survival prediction
and evaluate it using a confusion matrix and accuracy.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/titanic.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/titanic.csv was not found. Copy the Titanic CSV from Week 1 "
            "into the Week-2/data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "Survived" not in df.columns:
        raise ValueError("The Titanic dataset must contain the 'Survived' column.")

    # Keep useful passenger features for the assignment.
    selected_columns = [
        column
        for column in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        if column in df.columns
    ]

    X = df[selected_columns].copy()
    y = df["Survived"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\n=== Titanic Survival Prediction Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Titanic Logistic Regression - Confusion Matrix")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "titanic_confusion_matrix.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"\nSaved confusion matrix to: {chart_path}")


if __name__ == "__main__":
    main()
