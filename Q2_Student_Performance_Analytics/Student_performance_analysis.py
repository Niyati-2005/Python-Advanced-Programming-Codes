# %%
# ============================================================
# QUESTION 2: STUDENT PERFORMANCE ANALYTICS
# Advanced Python Programming
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# %%
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("student_performance.csv")

print("=" * 60)
print("STUDENT PERFORMANCE ANALYTICS")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# %%
# ============================================================
# 2. INITIAL DATASET ANALYSIS
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# %%
# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate records
duplicates_before = len(df)

df = df.drop_duplicates()

duplicates_removed = (
    duplicates_before - len(df)
)

print("\nDuplicate rows removed:",
      duplicates_removed)


# Convert numerical columns to numeric
numeric_columns = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows with missing values
rows_before = len(df)

df = df.dropna(
    subset=numeric_columns
)

rows_removed = rows_before - len(df)

print(
    "Rows removed due to missing values:",
    rows_removed
)


# %%
# ============================================================
# 4. CLEANED DATASET
# ============================================================

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFirst 5 Cleaned Rows:")
print(df.head())


# %%
# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

statistics = df[numeric_columns].describe()

print(
    statistics.round(2)
)


# %%
# ============================================================
# 6. ADDITIONAL STATISTICS
# ============================================================

print("\nMean:")
print(
    df[numeric_columns]
    .mean()
    .round(2)
)

print("\nMedian:")
print(
    df[numeric_columns]
    .median()
    .round(2)
)

print("\nStandard Deviation:")
print(
    df[numeric_columns]
    .std()
    .round(2)
)

print("\nMinimum:")
print(
    df[numeric_columns]
    .min()
    .round(2)
)

print("\nMaximum:")
print(
    df[numeric_columns]
    .max()
    .round(2)
)


# %%
# ============================================================
# 7. HISTOGRAM — STUDY HOURS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["weekly_self_study_hours"],
    bins=30,
    edgecolor="black"
)

plt.title(
    "Distribution of Weekly Self-Study Hours"
)

plt.xlabel(
    "Weekly Self-Study Hours"
)

plt.ylabel(
    "Number of Students"
)

plt.legend(
    ["Students"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 8. INTERPRETATION — STUDY HOURS
# ============================================================

print("\nINTERPRETATION — STUDY HOURS HISTOGRAM")

print(
    "The histogram shows the distribution of weekly "
    "self-study hours among the students."
)

print(
    "The highest bars represent the study-hour ranges "
    "containing the largest number of students."
)

print(
    "The distribution helps identify the typical study "
    "patterns within the dataset."
)


# %%
# ============================================================
# 9. HISTOGRAM — TOTAL SCORE
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["total_score"],
    bins=30,
    edgecolor="black"
)

plt.title(
    "Distribution of Student Total Scores"
)

plt.xlabel(
    "Total Score"
)

plt.ylabel(
    "Number of Students"
)

plt.legend(
    ["Students"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 10. INTERPRETATION — TOTAL SCORE
# ============================================================

print("\nINTERPRETATION — TOTAL SCORE HISTOGRAM")

print(
    "The histogram shows how the students' total scores "
    "are distributed."
)

print(
    "The concentration of observations indicates the "
    "most common score ranges."
)

print(
    "The spread of scores provides an overview of "
    "variation in student performance."
)


# %%
# ============================================================
# 11. BOXPLOT
# ============================================================

plt.figure(figsize=(11, 6))

df[numeric_columns].boxplot()

plt.title(
    "Boxplot of Student Performance Variables"
)

plt.xlabel(
    "Variables"
)

plt.ylabel(
    "Values"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 12. INTERPRETATION — BOXPLOT
# ============================================================

print("\nINTERPRETATION — BOXPLOT")

print(
    "The boxplot compares the distributions of study hours, "
    "attendance, class participation and total scores."
)

print(
    "The line inside each box represents the median, while "
    "the box represents the interquartile range."
)

print(
    "Values outside the whiskers may represent potential "
    "outliers in the dataset."
)


# %%
# ============================================================
# 13. STUDY HOURS VS TOTAL SCORE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="weekly_self_study_hours",
    y="total_score",
    alpha=0.2
)

plt.title(
    "Weekly Self-Study Hours vs Total Score"
)

plt.xlabel(
    "Weekly Self-Study Hours"
)

plt.ylabel(
    "Total Score"
)

plt.legend(
    ["Students"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 14. INTERPRETATION — STUDY HOURS VS SCORE
# ============================================================

correlation = df[
    [
        "weekly_self_study_hours",
        "total_score"
    ]
].corr().iloc[0, 1]

print("\nINTERPRETATION — STUDY HOURS VS TOTAL SCORE")

print(
    "The scatter plot shows the relationship between "
    "weekly self-study hours and total score."
)

print(
    f"The Pearson correlation coefficient is "
    f"{correlation:.2f}."
)

if correlation > 0:
    print(
        "The positive correlation indicates that higher "
        "study hours are generally associated with higher scores."
    )
elif correlation < 0:
    print(
        "The negative correlation indicates an inverse "
        "relationship between study hours and total score."
    )
else:
    print(
        "The correlation is close to zero, indicating "
        "little linear relationship between the variables."
    )


# %%
# ============================================================
# 15. PREPARE DATA FOR LINEAR REGRESSION
# ============================================================

# Independent variable
X = df[
    ["weekly_self_study_hours"]
]

# Dependent variable
y = df["total_score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:",
      len(X_train))

print(
    "Testing samples:",
    len(X_test)
)


# %%
# ============================================================
# 16. BUILD LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)


# %%
# ============================================================
# 17. REGRESSION RESULTS
# ============================================================

intercept = model.intercept_

coefficient = model.coef_[0]

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print(
    "\nIntercept:",
    round(intercept, 2)
)

print(
    "Coefficient:",
    round(coefficient, 2)
)

print("\nRegression Equation:")

print(
    f"Total Score = "
    f"{intercept:.2f} + "
    f"({coefficient:.2f} × Weekly Self-Study Hours)"
)


# %%
# ============================================================
# 18. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    "\nMean Absolute Error (MAE):",
    round(mae, 2)
)

print(
    "Mean Squared Error (MSE):",
    round(mse, 2)
)

print(
    "Root Mean Squared Error (RMSE):",
    round(rmse, 2)
)

print(
    "R² Score:",
    round(r2, 2)
)


# %%
# ============================================================
# 19. REGRESSION LINE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["weekly_self_study_hours"],
    y_test,
    alpha=0.2,
    label="Actual Scores"
)

# Sort values for regression line
sorted_indices = np.argsort(
    X_test["weekly_self_study_hours"].values
)

X_sorted = X_test.iloc[
    sorted_indices
]

y_sorted_pred = model.predict(
    X_sorted
)

plt.plot(
    X_sorted["weekly_self_study_hours"],
    y_sorted_pred,
    linewidth=2,
    label="Regression Line"
)

plt.title(
    "Weekly Self-Study Hours vs Total Score "
    "with Regression Line"
)

plt.xlabel(
    "Weekly Self-Study Hours"
)

plt.ylabel(
    "Total Score"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 20. ACTUAL VS PREDICTED MARKS
# ============================================================

comparison = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED SCORES")
print("=" * 60)

print(
    comparison.head(20).round(2)
)


# %%
# ============================================================
# 21. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3,
    label="Predictions"
)

# Perfect prediction reference line
minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    "Actual vs Predicted Student Scores"
)

plt.xlabel(
    "Actual Total Score"
)

plt.ylabel(
    "Predicted Total Score"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 22. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print(
    f"\nThe cleaned dataset contains {len(df)} student records."
)

print(
    f"The average weekly self-study time is "
    f"{df['weekly_self_study_hours'].mean():.2f} hours."
)

print(
    f"The average total score is "
    f"{df['total_score'].mean():.2f}."
)

print(
    f"The correlation between weekly self-study hours "
    f"and total score is {correlation:.2f}."
)

print(
    f"The linear regression model achieved an "
    f"R² score of {r2:.2f}."
)

print(
    "The actual-versus-predicted graph provides a visual "
    "comparison of the model's prediction performance."
)

print(
    "\nStudent performance analysis completed successfully."
)