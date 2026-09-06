# %%
# ============================================================
# QUESTION 4: WEATHER TREND ANALYSIS
# Advanced Python Programming
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# %%
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("weather_data.csv")

print("=" * 60)
print("WEATHER TREND ANALYSIS")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# %%
# ============================================================
# 2. INITIAL DATASET INFORMATION
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

# Convert Date_Time to datetime
df["Date_Time"] = pd.to_datetime(
    df["Date_Time"],
    errors="coerce"
)

# Convert numerical columns to numeric
numeric_columns = [
    "Temperature_C",
    "Humidity_pct",
    "Precipitation_mm",
    "Wind_Speed_kmh"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Count records before cleaning
rows_before = len(df)

# Remove duplicate records
df = df.drop_duplicates()

duplicates_removed = (
    rows_before - len(df)
)


# Remove rows containing missing values
rows_before_missing = len(df)

df = df.dropna(
    subset=[
        "Date_Time",
        "Temperature_C",
        "Humidity_pct",
        "Precipitation_mm",
        "Wind_Speed_kmh"
    ]
)

missing_rows_removed = (
    rows_before_missing - len(df)
)


print("\n" + "=" * 60)
print("DATA CLEANING RESULTS")
print("=" * 60)

print(
    "\nDuplicate rows removed:",
    duplicates_removed
)

print(
    "Rows removed because of missing values:",
    missing_rows_removed
)

print(
    "\nDate_Time data type:",
    df["Date_Time"].dtype
)

print(
    "\nCleaned dataset shape:",
    df.shape
)


# %%
# ============================================================
# 4. CREATE MONTH COLUMN
# ============================================================

df["Month"] = df["Date_Time"].dt.to_period("M")


# %%
# ============================================================
# 5. CALCULATE MONTHLY AVERAGES
# ============================================================

monthly_averages = (
    df.groupby("Month")
    .agg(
        Average_Temperature=(
            "Temperature_C",
            "mean"
        ),
        Average_Rainfall=(
            "Precipitation_mm",
            "mean"
        ),
        Average_Humidity=(
            "Humidity_pct",
            "mean"
        ),
        Average_Wind_Speed=(
            "Wind_Speed_kmh",
            "mean"
        )
    )
    .reset_index()
)

# Convert Period to Timestamp
monthly_averages["Month"] = (
    monthly_averages["Month"]
    .dt.to_timestamp()
)


print("\n" + "=" * 60)
print("MONTHLY WEATHER AVERAGES")
print("=" * 60)

print(
    monthly_averages.round(2)
)


# %%
# ============================================================
# 6. MONTHLY TEMPERATURE VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_averages["Month"],
    monthly_averages["Average_Temperature"],
    marker="o",
    label="Average Temperature"
)

plt.title(
    "Monthly Average Temperature"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 7. TEMPERATURE GRAPH INTERPRETATION
# ============================================================

highest_temp_month = monthly_averages.loc[
    monthly_averages["Average_Temperature"].idxmax()
]

lowest_temp_month = monthly_averages.loc[
    monthly_averages["Average_Temperature"].idxmin()
]

print("\nINTERPRETATION — MONTHLY TEMPERATURE")

print(
    "The line graph shows how average temperature "
    "changes across the observed months."
)

print(
    f"The highest monthly average temperature occurred "
    f"in {highest_temp_month['Month'].strftime('%B %Y')}."
)

print(
    f"The lowest monthly average temperature occurred "
    f"in {lowest_temp_month['Month'].strftime('%B %Y')}."
)

print(
    "The variations demonstrate the temperature trend "
    "and seasonal changes present in the dataset."
)


# %%
# ============================================================
# 8. MONTHLY RAINFALL VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.bar(
    monthly_averages["Month"],
    monthly_averages["Average_Rainfall"],
    width=20,
    label="Average Rainfall"
)

plt.title(
    "Monthly Average Rainfall"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Rainfall (mm)"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 9. RAINFALL GRAPH INTERPRETATION
# ============================================================

highest_rain_month = monthly_averages.loc[
    monthly_averages["Average_Rainfall"].idxmax()
]

print("\nINTERPRETATION — MONTHLY RAINFALL")

print(
    "The bar chart displays the average precipitation "
    "recorded during each month."
)

print(
    f"The highest average rainfall occurred in "
    f"{highest_rain_month['Month'].strftime('%B %Y')}."
)

print(
    "Differences between months indicate changes in "
    "precipitation patterns over time."
)


# %%
# ============================================================
# 10. Z-SCORE CALCULATION FOR TEMPERATURE
# ============================================================

temperature_mean = df["Temperature_C"].mean()

temperature_std = df["Temperature_C"].std()

df["Temperature_Z_Score"] = (
    (df["Temperature_C"] - temperature_mean)
    / temperature_std
)


# Identify unusually hot days
# Z-score > 2 means temperature is more than
# two standard deviations above the mean.

hot_days = df[
    df["Temperature_Z_Score"] > 2
]


print("\n" + "=" * 60)
print("UNUSUALLY HOT DAYS — Z-SCORE METHOD")
print("=" * 60)

print(
    f"\nMean temperature: "
    f"{temperature_mean:.2f} °C"
)

print(
    f"Standard deviation: "
    f"{temperature_std:.2f} °C"
)

print(
    f"\nNumber of unusually hot observations: "
    f"{len(hot_days)}"
)

print(
    f"Percentage of unusually hot observations: "
    f"{(len(hot_days) / len(df)) * 100:.2f}%"
)

print("\nSample of unusually hot observations:")

print(
    hot_days[
        [
            "Location",
            "Date_Time",
            "Temperature_C",
            "Temperature_Z_Score"
        ]
    ]
    .sort_values(
        "Temperature_C",
        ascending=False
    )
    .head(20)
)


# %%
# ============================================================
# 11. UNUSUALLY HOT TEMPERATURE VISUALIZATION
# ============================================================

# To keep the graph readable with 1 million observations,
# sample a maximum of 10,000 normal observations.

normal_days = df[
    df["Temperature_Z_Score"] <= 2
]

normal_sample = normal_days.sample(
    n=min(10000, len(normal_days)),
    random_state=42
)

hot_sample = hot_days.sample(
    n=min(5000, len(hot_days)),
    random_state=42
)

plt.figure(figsize=(12, 6))

plt.scatter(
    normal_sample["Date_Time"],
    normal_sample["Temperature_C"],
    alpha=0.25,
    label="Normal Temperature"
)

plt.scatter(
    hot_sample["Date_Time"],
    hot_sample["Temperature_C"],
    alpha=0.7,
    label="Unusually Hot"
)

plt.title(
    "Temperature Observations and Unusually Hot Days"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 12. HOT DAYS GRAPH INTERPRETATION
# ============================================================

print("\nINTERPRETATION — UNUSUALLY HOT DAYS")

print(
    "The scatter plot displays temperature observations "
    "over time and highlights unusually hot observations."
)

print(
    "A temperature Z-score greater than 2 was used as "
    "the threshold for identifying unusually hot observations."
)

print(
    f"The analysis identified {len(hot_days)} "
    "observations above this threshold."
)


# %%
# ============================================================
# 13. PREPARE DATA FOR LINEAR REGRESSION
# ============================================================

# Independent variables
X = df[
    [
        "Humidity_pct",
        "Precipitation_mm",
        "Wind_Speed_kmh"
    ]
]

# Dependent variable
y = df[
    "Temperature_C"
]


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION DATA")
print("=" * 60)

print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# %%
# ============================================================
# 14. BUILD LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# Predict temperature
y_pred = model.predict(
    X_test
)


# %%
# ============================================================
# 15. MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL")
print("=" * 60)

print(
    "\nIntercept:",
    round(model.intercept_, 4)
)

print("\nCoefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):
    print(
        f"{feature}: {coefficient:.4f}"
    )


# %%
# ============================================================
# 16. MODEL EVALUATION
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nR² Score: {r2:.4f}"
)

print(
    f"Mean Squared Error (MSE): {mse:.4f}"
)


# %%
# ============================================================
# 17. ACTUAL VS PREDICTED TEMPERATURE
# ============================================================

# Sample points for visualization because the dataset
# contains 1 million observations.

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

plot_sample = comparison.sample(
    n=min(5000, len(comparison)),
    random_state=42
)


plt.figure(figsize=(10, 6))

plt.scatter(
    plot_sample["Actual"],
    plot_sample["Predicted"],
    alpha=0.3,
    label="Predictions"
)


# Perfect prediction line
minimum = min(
    plot_sample["Actual"].min(),
    plot_sample["Predicted"].min()
)

maximum = max(
    plot_sample["Actual"].max(),
    plot_sample["Predicted"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    "Actual vs Predicted Temperature"
)

plt.xlabel(
    "Actual Temperature (°C)"
)

plt.ylabel(
    "Predicted Temperature (°C)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 18. ACTUAL VS PREDICTED INTERPRETATION
# ============================================================

print("\nINTERPRETATION — ACTUAL VS PREDICTED")

print(
    "The scatter plot compares the actual temperatures "
    "with temperatures predicted by the Linear Regression model."
)

print(
    "Points closer to the diagonal reference line represent "
    "more accurate predictions."
)

print(
    f"The model achieved an R² score of {r2:.4f}, "
    f"while the MSE was {mse:.4f}."
)


# %%
# ============================================================
# 19. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print(
    f"\nThe cleaned weather dataset contains "
    f"{len(df):,} observations."
)

print(
    f"The average temperature is "
    f"{df['Temperature_C'].mean():.2f} °C."
)

print(
    f"The average humidity is "
    f"{df['Humidity_pct'].mean():.2f}%."
)

print(
    f"The average rainfall is "
    f"{df['Precipitation_mm'].mean():.2f} mm."
)

print(
    f"The Z-score method identified "
    f"{len(hot_days):,} unusually hot observations."
)

print(
    f"The Linear Regression model achieved an "
    f"R² score of {r2:.4f}."
)

print(
    f"The Mean Squared Error of the model is "
    f"{mse:.4f}."
)

print(
    "\nWeather trend analysis completed successfully."
)