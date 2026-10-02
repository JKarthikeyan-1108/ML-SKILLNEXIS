"""
SkillNexis - Week 1 Assignments
ML Fundamentals + Data Preprocessing

Assignments covered:
1. Load a dataset using Pandas and summarize basic statistics.
2. Handle missing data using mean/median imputation.
3. Encode categorical variables using LabelEncoder and OneHotEncoder.
"""

from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/titanic.csv")


def load_dataset():
    """Load the Titanic dataset from data/titanic.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/titanic.csv was not found. Download the Kaggle Titanic "
            "dataset and place the CSV file at data/titanic.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n=== Dataset Preview ===")
    print(df.head())

    print("\n=== Dataset Information ===")
    df.info()

    print("\n=== Descriptive Statistics ===")
    print(df.describe(include="all"))

    return df


def assignment_2_missing_data(df):
    """Demonstrate mean and median imputation."""
    result = df.copy()

    if "Age" in result.columns:
        mean_imputer = SimpleImputer(strategy="mean")
        median_imputer = SimpleImputer(strategy="median")

        mean_df = result.copy()
        median_df = result.copy()

        mean_df[["Age"]] = mean_imputer.fit_transform(mean_df[["Age"]])
        median_df[["Age"]] = median_imputer.fit_transform(median_df[["Age"]])

        print("\n=== Assignment 2: Missing Age Values ===")
        print("Original missing Age values :", result["Age"].isna().sum())
        print("After mean imputation      :", mean_df["Age"].isna().sum())
        print("After median imputation   :", median_df["Age"].isna().sum())

        print("\nMean-imputed Age sample:")
        print(mean_df[["Age"]].head())

        print("\nMedian-imputed Age sample:")
        print(median_df[["Age"]].head())

    return result


def assignment_3_encoding(df):
    """Demonstrate LabelEncoder and OneHotEncoder."""
    print("\n=== Assignment 3: Categorical Encoding ===")

    # LabelEncoder for Sex
    if "Sex" in df.columns:
        label_encoder = LabelEncoder()
        sex_clean = df["Sex"].fillna(df["Sex"].mode()[0]).astype(str)
        df["Sex_LabelEncoded"] = label_encoder.fit_transform(sex_clean)

        print("\nLabelEncoder classes for Sex:")
        print(dict(enumerate(label_encoder.classes_)))
        print(df[["Sex", "Sex_LabelEncoded"]].head())

    # OneHotEncoder for Embarked
    if "Embarked" in df.columns:
        embarked_clean = df["Embarked"].fillna(df["Embarked"].mode()[0]).astype(str)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(embarked_clean.to_frame())

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(["Embarked"]),
            index=df.index,
        )

        print("\nOneHotEncoder output for Embarked:")
        print(encoded_df.head())

    return df


def train_test_split_demo(df):
    """Demonstrate a basic train/test split using the target Survived."""
    if "Survived" not in df.columns:
        return

    features = df.drop(columns=["Survived"])
    target = df["Survived"]

    # Use only numeric columns for this simple demonstration.
    numeric_features = features.select_dtypes(include="number").copy()
    numeric_features = numeric_features.fillna(numeric_features.median())

    X_train, X_test, y_train, y_test = train_test_split(
        numeric_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    print("\n=== Train/Test Split ===")
    print("Training features:", X_train.shape)
    print("Testing features :", X_test.shape)
    print("Training target  :", y_train.shape)
    print("Testing target   :", y_test.shape)


if __name__ == "__main__":
    data = load_dataset()
    assignment_2_missing_data(data)
    assignment_3_encoding(data)
    train_test_split_demo(data)
    print("\nWeek 1 assignments completed.")
