"""
SkillNexis - Week 2 Mini Project
House Price Prediction Model

Tasks:
- Train Linear Regression model
- Predict house prices
- Evaluate R² score
- Plot predicted vs actual values

Dataset:
Kaggle House Prices Dataset (train.csv)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/train.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/train.csv was not found. Download the Kaggle House Prices "
            "train.csv file and place it at data/train.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nDataset information:")
    df.info()

    if "SalePrice" not in df.columns:
        raise ValueError("The dataset must contain the 'SalePrice' target column.")

    # Separate features and target.
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Drop Id because it is an identifier rather than a useful predictive feature.
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTraining Linear Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=== House Price Prediction Results ===")
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    results = pd.DataFrame(
        {
            "ActualPrice": y_test.values,
            "PredictedPrice": predictions,
        }
    )

    print("\nSample predictions:")
    print(results.head(10))

    results.to_csv(OUTPUT_DIR / "house_price_predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Predicted vs Actual House Prices")

    min_price = min(y_test.min(), predictions.min())
    max_price = max(y_test.max(), predictions.max())
    plt.plot([min_price, max_price], [min_price, max_price], linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print("- output/house_price_predictions.csv")
    print("- output/predicted_vs_actual.png")


if __name__ == "__main__":
    main()
