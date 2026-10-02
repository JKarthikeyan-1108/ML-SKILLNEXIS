"""
SkillNexis - Week 1 Mini Project
Titanic Survival Prediction - Data Cleaning Project

Tasks:
- Clean missing data
- Encode Sex and Embarked
- Visualize age distribution
- Export cleaned dataset as a new CSV
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA_PATH = Path("data/titanic.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_titanic_data(df):
    df = df.copy()

    # Fill Age using median.
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill Embarked using mode.
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fill Fare if present.
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Drop Cabin because it contains a large amount of missing data
    # and the Week 1 task focuses on Sex and Embarked encoding.
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])

    return df


def encode_columns(df):
    df = df.copy()

    # LabelEncoder for Sex.
    if "Sex" in df.columns:
        label_encoder = LabelEncoder()
        df["Sex"] = label_encoder.fit_transform(df["Sex"].astype(str))

    # OneHotEncoder for Embarked.
    if "Embarked" in df.columns:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded = encoder.fit_transform(df[["Embarked"]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(["Embarked"]),
            index=df.index,
        )

        df = pd.concat(
            [df.drop(columns=["Embarked"]), encoded_df],
            axis=1,
        )

    return df


def visualize_age_distribution(df):
    if "Age" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=30, kde=True)
    plt.title("Titanic Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "age_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Age distribution saved to: {chart_path}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/titanic.csv was not found. Download the Kaggle Titanic "
            "dataset and place it at data/titanic.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    cleaned = clean_titanic_data(df)
    cleaned = encode_columns(cleaned)

    print("\nCleaned shape:", cleaned.shape)
    print("\nMissing values after cleaning:")
    print(cleaned.isnull().sum())

    # Age visualization is based on the cleaned numeric Age column.
    visualize_age_distribution(cleaned)

    output_path = OUTPUT_DIR / "cleaned_titanic.csv"
    cleaned.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")
    print("\nFirst five rows of cleaned dataset:")
    print(cleaned.head())


if __name__ == "__main__":
    main()
