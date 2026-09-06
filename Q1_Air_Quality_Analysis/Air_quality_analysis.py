# %%
# ============================================================
# QUESTION 1: AIR QUALITY DATA ANALYSIS
# Advanced Python Programming
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("city_day.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# %%
# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("Dataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# %%
# ============================================================
# 3. DATA CLEANING
# ============================================================

# Convert Date to datetime format
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Required pollutants
pollutants = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO"
]

# Convert pollutant columns to numeric
for column in pollutants:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove duplicate rows
duplicates_before = len(df)

df = df.drop_duplicates()

duplicates_removed = (
    duplicates_before - len(df)
)

# Remove rows where City or Date is missing
rows_before = len(df)

df = df.dropna(
    subset=["City", "Date"]
)

rows_removed = rows_before - len(df)

# Create Year and Month columns
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

print("Data cleaning completed.")

print("\nDuplicate rows removed:",
      duplicates_removed)

print("Rows removed due to missing City/Date:",
      rows_removed)

print("\nCleaned dataset shape:")
print(df.shape)


# %%
# ============================================================
# 4. CITY INFORMATION
# ============================================================

number_of_cities = df["City"].nunique()

cities = df["City"].unique()

print("Number of cities:")
print(number_of_cities)

print("\nCities:")
print(cities)


# %%
# ============================================================
# 5. AVERAGE PM2.5 BY CITY
# ============================================================

city_pm25 = (
    df.groupby("City")["PM2.5"]
      .mean()
      .sort_values(ascending=False)
)

print("Average PM2.5 by City:")
print(city_pm25.round(2))


# %%
# ============================================================
# 6. MOST POLLUTED CITY
# ============================================================

most_polluted_city = city_pm25.idxmax()

highest_pm25 = city_pm25.max()

print("Most Polluted City:")
print(most_polluted_city)

print("\nHighest Average PM2.5:")
print(round(highest_pm25, 2))


# %%
# ============================================================
# 7. GRAPH 1 — AVERAGE PM2.5 BY CITY
# ============================================================

plt.figure(figsize=(14, 7))

city_pm25.plot(
    kind="bar"
)

plt.title(
    "Average PM2.5 Concentration by City"
)

plt.xlabel("City")

plt.ylabel(
    "Average PM2.5 Concentration"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 8. MONTHLY AVERAGE PM2.5
# ============================================================

monthly_pm25 = (
    df.groupby(
        ["City", "Year", "Month"]
    )["PM2.5"]
    .mean()
    .reset_index()
)

# Create Year-Month date
monthly_pm25["Year_Month"] = pd.to_datetime(
    monthly_pm25["Year"].astype(str)
    + "-"
    + monthly_pm25["Month"].astype(str)
    + "-01"
)

# Round values
monthly_pm25["PM2.5"] = (
    monthly_pm25["PM2.5"].round(2)
)

print("Monthly Average PM2.5:")
print(monthly_pm25.head(20))


# %%
# ============================================================
# 9. SELECT TOP 5 CITIES DYNAMICALLY
# ============================================================

top_cities = city_pm25.head(5).index

monthly_top_cities = monthly_pm25[
    monthly_pm25["City"].isin(top_cities)
]

print("Top 5 most polluted cities:")
print(top_cities.tolist())


# %%
# ============================================================
# 10. GRAPH 2 — COMPARATIVE MONTHLY PM2.5
# ============================================================

plt.figure(figsize=(15, 8))

sns.lineplot(
    data=monthly_top_cities,
    x="Year_Month",
    y="PM2.5",
    hue="City",
    marker="o"
)

plt.title(
    "Monthly Average PM2.5 Levels "
    "Across the Five Most Polluted Cities"
)

plt.xlabel("Month")

plt.ylabel(
    "Average PM2.5 Concentration"
)

plt.legend(
    title="City",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

correlation_matrix = df[pollutants].corr()

print("Correlation Matrix:")
print(
    correlation_matrix.round(2)
)


# %%
# ============================================================
# 12. GRAPH 3 — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Between Major Air Pollutants"
)

plt.xlabel("Pollutants")

plt.ylabel("Pollutants")

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 13. FIND STRONGEST CORRELATION
# ============================================================

# Copy matrix so that diagonal values can be ignored
correlation_without_diagonal = (
    correlation_matrix.copy()
)

# Remove self-correlations
np.fill_diagonal(
    correlation_without_diagonal.values,
    np.nan
)

# Find strongest absolute correlation
strongest_pair = (
    correlation_without_diagonal
    .abs()
    .stack()
    .idxmax()
)

strongest_value = (
    correlation_matrix.loc[
        strongest_pair[0],
        strongest_pair[1]
    ]
)

print("Strongest correlation:")
print(
    strongest_pair[0],
    "and",
    strongest_pair[1]
)

print("Correlation value:")
print(round(strongest_value, 2))


# %%
# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    "\nNumber of cities:",
    number_of_cities
)

print(
    "\nMost polluted city:",
    most_polluted_city
)

print(
    "Highest average PM2.5:",
    round(highest_pm25, 2)
)

print(
    "\nStrongest pollutant correlation:",
    strongest_pair[0],
    "and",
    strongest_pair[1]
)

print(
    "Correlation:",
    round(strongest_value, 2)
)


# %%
# ============================================================
# 15. CONCLUSION
# ============================================================

print("\nCONCLUSION:")

print(
    f"The analysis was performed on air quality data "
    f"from {number_of_cities} Indian cities."
)

print(
    f"{most_polluted_city} recorded the highest "
    f"average PM2.5 concentration of "
    f"{highest_pm25:.2f}."
)

print(
    "Monthly analysis showed variation in PM2.5 "
    "concentrations across different cities and months."
)

print(
    f"The strongest correlation was observed between "
    f"{strongest_pair[0]} and {strongest_pair[1]}, "
    f"with a correlation coefficient of "
    f"{strongest_value:.2f}."
)

print(
    "Overall, the dataset demonstrates significant "
    "spatial and temporal variation in air pollution."
)