# Week 2 – Supervised Learning (Regression & Classification)

Week 2 focuses on model building, training, and evaluation.

## Topics
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- MSE, R², and Confusion Matrix

## Assignments

1. Build a Linear Regression model on a housing dataset to predict price.
2. Train Logistic Regression on the Titanic dataset for survival prediction.

## Mini Project 2 – House Price Prediction Model

Dataset: Kaggle – House Prices Dataset

Tasks:
- Train Linear Regression model
- Predict house prices
- Evaluate R² score
- Plot predicted vs actual values

## Project structure

```text
Week-2/
├── README.md
├── requirements.txt
├── linear_regression_house_prices.py
├── logistic_regression_titanic.py
├── data/
│   ├── train.csv
│   └── titanic.csv
└── output/
    └── predicted_vs_actual.png
```

## Dataset placement

For the Kaggle House Prices dataset, place `train.csv` in:

```text
data/train.csv
```

For the Titanic assignment, place `titanic.csv` in:

```text
data/titanic.csv
```

You can reuse the Titanic CSV from Week 1 by copying it into this Week-2 `data` folder.

## Run

```bash
pip install -r requirements.txt
python linear_regression_house_prices.py
python logistic_regression_titanic.py
```
