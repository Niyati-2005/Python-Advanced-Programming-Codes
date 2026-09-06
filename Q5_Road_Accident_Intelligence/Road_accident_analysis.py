# ============================================================
# QUESTION 5 - ROAD ACCIDENT INTELLIGENCE DASHBOARD
# Advanced Python Programming
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Change this filename if your CSV has a different name
df = pd.read_csv("road_accident_dataset.csv")

print("=" * 70)
print("ROAD ACCIDENT INTELLIGENCE DASHBOARD")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 3. CREATE DATE COLUMN
# ============================================================

# The dataset has Year and Month separately.
# Combine them to create a proper date column.

df["Date"] = pd.to_datetime(
    df["Month"] + " " + df["Year"].astype(str),
    format="%B %Y",
    errors="coerce"
)

# Remove rows where date conversion failed
df = df.dropna(subset=["Date"])

# Sort chronologically
df = df.sort_values("Date")

print("\nDate conversion completed.")

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print("\nAccident statistics:")

print(
    "Total accidents:",
    len(df)
)

print(
    "Total injuries:",
    df["Number of Injuries"].sum()
)

print(
    "Total fatalities:",
    df["Number of Fatalities"].sum()
)

print(
    "Average injuries per accident:",
    round(
        df["Number of Injuries"].mean(),
        2
    )
)

print(
    "Average fatalities per accident:",
    round(
        df["Number of Fatalities"].mean(),
        2
    )
)


# ============================================================
# 5. MONTHLY ACCIDENT STATISTICS
# ============================================================

monthly_stats = (
    df.groupby("Date")
    .agg(
        Accidents=("Date", "count"),
        Injuries=("Number of Injuries", "sum"),
        Fatalities=("Number of Fatalities", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("MONTHLY ACCIDENT STATISTICS")
print("=" * 70)

print(monthly_stats)


# ============================================================
# GRAPH 1 - MONTHLY ACCIDENT TREND
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_stats["Date"],
    monthly_stats["Accidents"],
    marker="o",
    label="Monthly Accidents"
)

plt.title(
    "Monthly Road Accident Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Number of Accidents"
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


# ============================================================
# INTERPRETATION - GRAPH 1
# ============================================================

highest_month = monthly_stats.loc[
    monthly_stats["Accidents"].idxmax()
]

lowest_month = monthly_stats.loc[
    monthly_stats["Accidents"].idxmin()
]

print("\nINTERPRETATION - MONTHLY ACCIDENT TREND")

print(
    "The graph shows how the number of recorded "
    "accidents changes over time."
)

print(
    f"The highest accident count occurred in "
    f"{highest_month['Date'].strftime('%B %Y')}."
)

print(
    f"The lowest accident count occurred in "
    f"{lowest_month['Date'].strftime('%B %Y')}."
)

print(
    "The variations indicate changes in accident frequency "
    "across the observation period."
)


# ============================================================
# 6. TOP 5 COUNTRIES
# ============================================================

top_5_countries = (
    df["Country"]
    .value_counts()
    .head(5)
)

print("\n" + "=" * 70)
print("TOP 5 COUNTRIES BY ACCIDENT COUNT")
print("=" * 70)

print(top_5_countries)


# ============================================================
# GRAPH 2 - TOP 5 COUNTRIES
# ============================================================

plt.figure(figsize=(10, 6))

top_5_countries.plot(
    kind="bar"
)

plt.title(
    "Top 5 Countries by Number of Accidents"
)

plt.xlabel(
    "Country"
)

plt.ylabel(
    "Number of Accidents"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# INTERPRETATION - GRAPH 2
# ============================================================

print("\nINTERPRETATION - TOP 5 COUNTRIES")

print(
    "The bar chart compares accident counts among "
    "the five countries with the highest number of records."
)

print(
    f"{top_5_countries.index[0]} has the highest "
    "number of recorded accidents."
)

print(
    "The comparison helps identify geographical regions "
    "with relatively higher accident frequencies."
)


# ============================================================
# 7. STACKED BAR CHART - ROAD TYPE
# ============================================================

# Select top 5 countries
top_country_df = df[
    df["Country"].isin(
        top_5_countries.index
    )
].copy()

# Create country × road type table
road_type_table = pd.crosstab(
    top_country_df["Country"],
    top_country_df["Road Type"]
)

# Keep top 5 countries in the same order
road_type_table = road_type_table.reindex(
    top_5_countries.index
)

# Keep the most common road types
common_road_types = (
    top_country_df["Road Type"]
    .value_counts()
    .head(6)
    .index
)

road_type_table = road_type_table[
    [
        column
        for column in road_type_table.columns
        if column in common_road_types
    ]
]


# ============================================================
# GRAPH 3 - STACKED BAR CHART
# ============================================================

road_type_table.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

plt.title(
    "Accidents by Road Type - Top 5 Countries"
)

plt.xlabel(
    "Country"
)

plt.ylabel(
    "Number of Accidents"
)

plt.legend(
    title="Road Type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# INTERPRETATION - GRAPH 3
# ============================================================

print("\nINTERPRETATION - STACKED BAR CHART")

print(
    "The stacked bar chart compares accident counts "
    "by road type across the five leading countries."
)

print(
    "Each section of a bar represents a different road type."
)

print(
    "This allows differences in road-related accident "
    "patterns between countries to be compared."
)


# ============================================================
# 8. WEATHER CONDITION HEATMAP
# ============================================================

# Select top 8 weather conditions
common_weather = (
    df["Weather Conditions"]
    .value_counts()
    .head(8)
    .index
)

weather_df = df[
    df["Country"].isin(
        top_5_countries.index
    )
    &
    df["Weather Conditions"].isin(
        common_weather
    )
]

weather_table = pd.crosstab(
    weather_df["Country"],
    weather_df["Weather Conditions"]
)

weather_table = weather_table.reindex(
    top_5_countries.index
)


# ============================================================
# GRAPH 4 - HEATMAP
# ============================================================

plt.figure(figsize=(12, 7))

sns.heatmap(
    weather_table,
    annot=True,
    fmt="d",
    cmap="YlOrRd"
)

plt.title(
    "Accidents by Country and Weather Condition"
)

plt.xlabel(
    "Weather Conditions"
)

plt.ylabel(
    "Country"
)

plt.tight_layout()

plt.show()


# ============================================================
# INTERPRETATION - GRAPH 4
# ============================================================

print("\nINTERPRETATION - WEATHER HEATMAP")

print(
    "The heatmap shows the distribution of accidents "
    "under different weather conditions."
)

print(
    "Darker cells represent a higher number of "
    "recorded accidents."
)

print(
    "The visualization helps identify weather conditions "
    "associated with relatively higher accident frequencies."
)


# ============================================================
# 9. ACCIDENT SPIKE DETECTION
# ============================================================

# Calculate 3-month rolling average
monthly_stats["Rolling_Average"] = (
    monthly_stats["Accidents"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
)

# Define an accident spike as:
# Actual accidents > 120% of rolling average

monthly_stats["Accident_Spike"] = (
    monthly_stats["Accidents"]
    >
    monthly_stats["Rolling_Average"] * 1.20
)

spikes = monthly_stats[
    monthly_stats["Accident_Spike"]
]


print("\n" + "=" * 70)
print("ACCIDENT SPIKE DETECTION")
print("=" * 70)

print(
    "\nNumber of potential accident spikes:",
    len(spikes)
)

print("\nPotential spike months:")

print(
    spikes[
        [
            "Date",
            "Accidents",
            "Rolling_Average"
        ]
    ].round(2)
)


# ============================================================
# GRAPH 5 - ACCIDENT SPIKES
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_stats["Date"],
    monthly_stats["Accidents"],
    marker="o",
    label="Actual Accidents"
)

plt.plot(
    monthly_stats["Date"],
    monthly_stats["Rolling_Average"],
    linestyle="--",
    label="3-Month Rolling Average"
)

plt.scatter(
    spikes["Date"],
    spikes["Accidents"],
    s=80,
    label="Potential Accident Spike"
)

plt.title(
    "Accident Trend and Potential Spikes"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Number of Accidents"
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


# ============================================================
# INTERPRETATION - GRAPH 5
# ============================================================

print("\nINTERPRETATION - ACCIDENT SPIKES")

print(
    "The graph compares actual monthly accident counts "
    "with a three-month rolling average."
)

print(
    "Potential spikes are identified when the actual count "
    "is more than 20% above the rolling average."
)

print(
    f"The method detected {len(spikes)} potential "
    "accident spike months."
)


# ============================================================
# 10. PREPARE DATA FOR LINEAR REGRESSION
# ============================================================

# Create numerical time index
monthly_stats["Time_Index"] = np.arange(
    len(monthly_stats)
)

# Features
X = monthly_stats[
    [
        "Time_Index",
        "Injuries",
        "Fatalities"
    ]
]

# Target
y = monthly_stats[
    "Accidents"
]


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

# Chronological split
# First 80% -> training
# Last 20% -> testing

split_index = int(
    len(monthly_stats) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ============================================================
# 12. LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)

print(
    f"\nR² Score: {r2:.4f}"
)

print(
    f"Mean Squared Error: {mse:.4f}"
)


# ============================================================
# 14. GRAPH 6 - ACTUAL VS PREDICTED
# ============================================================

test_dates = monthly_stats[
    "Date"
].iloc[
    split_index:
]

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    y_test,
    marker="o",
    label="Actual Accidents"
)

plt.plot(
    test_dates,
    y_pred,
    marker="o",
    linestyle="--",
    label="Predicted Accidents"
)

plt.title(
    "Actual vs Predicted Monthly Accidents"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Number of Accidents"
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


# ============================================================
# INTERPRETATION - GRAPH 6
# ============================================================

print("\nINTERPRETATION - ACTUAL VS PREDICTED")

print(
    "The graph compares the actual accident counts "
    "with the values predicted by the Linear Regression model."
)

print(
    "A smaller difference between the two lines indicates "
    "better predictive performance."
)

print(
    f"The model produced an R² score of {r2:.4f}."
)

print(
    f"The Mean Squared Error was {mse:.4f}."
)


# ============================================================
# 15. FUTURE REVENUE-STYLE FORECAST
# ============================================================
# For this question we forecast future ACCIDENT COUNTS.

# Number of future months
future_months = 6

last_date = monthly_stats["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=future_months,
    freq="MS"
)

# Future time index
future_time_index = np.arange(
    len(monthly_stats),
    len(monthly_stats) + future_months
)

# Use average injuries and fatalities
# as simple future feature estimates.
average_injuries = monthly_stats[
    "Injuries"
].mean()

average_fatalities = monthly_stats[
    "Fatalities"
].mean()

future_data = pd.DataFrame(
    {
        "Time_Index": future_time_index,
        "Injuries": average_injuries,
        "Fatalities": average_fatalities
    }
)

future_predictions = model.predict(
    future_data
)


# ============================================================
# GRAPH 7 - FUTURE ACCIDENT FORECAST
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_stats["Date"],
    monthly_stats["Accidents"],
    marker="o",
    label="Historical Accidents"
)

plt.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Forecasted Accidents"
)

plt.title(
    "Future Road Accident Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Number of Accidents"
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


# ============================================================
# INTERPRETATION - GRAPH 7
# ============================================================

print("\nINTERPRETATION - FUTURE FORECAST")

print(
    "The forecast extends the Linear Regression model "
    "into the next six months."
)

print(
    "The predicted values provide an estimate of future "
    "monthly accident frequency based on the model."
)

print(
    "These forecasts should be treated as estimates rather "
    "than exact future accident counts."
)


# ============================================================
# 16. FUTURE FORECAST TABLE
# ============================================================

forecast_table = pd.DataFrame(
    {
        "Month": future_dates,
        "Predicted Accidents": np.round(
            future_predictions,
            0
        ).astype(int)
    }
)

print("\n" + "=" * 70)
print("FUTURE ACCIDENT FORECAST")
print("=" * 70)

print(
    forecast_table
)


# ============================================================
# 17. FINAL DASHBOARD SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ROAD ACCIDENT DASHBOARD SUMMARY")
print("=" * 70)

print(
    f"\nTotal records analysed: {len(df):,}"
)

print(
    f"Total injuries: "
    f"{df['Number of Injuries'].sum():,}"
)

print(
    f"Total fatalities: "
    f"{df['Number of Fatalities'].sum():,}"
)

print(
    f"Top country: "
    f"{top_5_countries.index[0]}"
)

print(
    f"Accidents in top country: "
    f"{top_5_countries.iloc[0]:,}"
)

print(
    f"Potential accident spikes: "
    f"{len(spikes)}"
)

print(
    f"Linear Regression R²: "
    f"{r2:.4f}"
)

print(
    f"Linear Regression MSE: "
    f"{mse:.4f}"
)


# ============================================================
# 18. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

print(
    "\nThe road accident dataset was cleaned and analysed "
    "to identify temporal and geographical accident patterns."
)

print(
    "Monthly accident statistics were calculated along with "
    "injury and fatality information."
)

print(
    f"The analysis identified {top_5_countries.index[0]} "
    "as the country with the highest number of recorded accidents."
)

print(
    "Road type and weather conditions were analysed using "
    "comparative charts and a heatmap."
)

print(
    f"Rolling-average analysis identified {len(spikes)} "
    "potential accident spike periods."
)

print(
    "Finally, a Linear Regression model was used to predict "
    "monthly accident counts and generate a six-month forecast."
)

print(
    f"The model achieved an R² score of {r2:.4f} "
    f"and an MSE of {mse:.4f}."
)

print(
    "\nRoad Accident Intelligence Dashboard completed successfully."
)