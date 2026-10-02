# Week 1 – ML Fundamentals + Data Preprocessing

This folder contains the Week 1 assignments for the SkillNexis Machine Learning & AI course.

## Topics covered
- Machine Learning fundamentals
- Importing and exploring datasets
- Missing-data handling using mean/median imputation
- Categorical encoding with `LabelEncoder` and `OneHotEncoder`
- Train/Test Split
- Titanic Survival Prediction – Data Cleaning Project

## Assignments

1. **Load and explore a dataset using Pandas**
   - `.info()`
   - `.describe()`

2. **Handle missing data**
   - Mean imputation
   - Median imputation

3. **Encode categorical variables**
   - `LabelEncoder`
   - `OneHotEncoder`

4. **Mini Project – Titanic Survival Prediction: Data Cleaning**
   - Clean missing data
   - Encode `Sex` and `Embarked`
   - Visualize age distribution
   - Export the cleaned dataset as CSV

## Project structure

```text
Week-1/
├── README.md
├── requirements.txt
├── week1_assignments.py
├── titanic_cleaning.py
├── data/
│   └── titanic.csv          # Add the Kaggle Titanic dataset here
└── output/
    ├── cleaned_titanic.csv
    └── age_distribution.png
```

## Setup

```bash
pip install -r requirements.txt
```

Place the Kaggle Titanic CSV file at:

```text
data/titanic.csv
```

Then run:

```bash
python week1_assignments.py
python titanic_cleaning.py
```

The cleaned dataset and age-distribution chart will be saved in the `output/` folder.

> The assignment sheet specifies the Titanic Dataset (Kaggle) as the source dataset.
