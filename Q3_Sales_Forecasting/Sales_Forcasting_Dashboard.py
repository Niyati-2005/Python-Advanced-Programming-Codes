# %%
# ============================================================
# QUESTION 3: SALES FORECASTING DASHBOARD
# Advanced Python Programming
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %%
# ============================================================
# 1. LOAD EXCEL DATASET
# ============================================================

df = pd.read_excel("Retail Sales Data Project.xlsx")

print("=" * 60)
print("RETAIL SALES ANALYSIS")
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
# 3. DATA CLEANING AND DATE CONVERSION
# ============================================================

# Convert Date column to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Convert numerical columns to numeric
numeric_columns = [
    "Quantity",
    "Price per Unit",
    "Total Amount"
]

for column in numeric_columns:
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


# Remove rows with missing important values
rows_before = len(df)

df = df.dropna(
    subset=[
        "Date",
        "Product Category",
        "Quantity",
        "Price per Unit",
        "Total Amount"
    ]
)

missing_rows_removed = (
    rows_before - len(df)
)


print("\nData Cleaning Completed")

print(
    "Duplicate rows removed:",
    duplicates_removed
)

print(
    "Rows removed due to missing values:",
    missing_rows_removed
)

print(
    "\nDate data type after conversion:",
    df["Date"].dtype
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
print("SALES STATISTICS")
print("=" * 60)

print(
    df[
        [
            "Quantity",
            "Price per Unit",
            "Total Amount"
        ]
    ].describe().round(2)
)


# %%
# ============================================================
# 6. CALCULATE MONTHLY REVENUE
# ============================================================

# Create a Month column
df["Month"] = df["Date"].dt.to_period("M")

# Calculate monthly revenue
monthly_revenue = (
    df.groupby("Month")["Total Amount"]
    .sum()
    .reset_index()
)

# Convert Period back to timestamp for plotting
monthly_revenue["Month"] = (
    monthly_revenue["Month"]
    .dt.to_timestamp()
)

print("\n" + "=" * 60)
print("MONTHLY REVENUE")
print("=" * 60)

print(
    monthly_revenue
)


# %%
# ============================================================
# 7. BEST-SELLING CATEGORY
# ============================================================

category_sales = (
    df.groupby("Product Category")
    .agg(
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Total Amount", "sum")
    )
    .sort_values(
        "Total_Quantity",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("PRODUCT CATEGORY PERFORMANCE")
print("=" * 60)

print(
    category_sales
)


best_category = category_sales[
    "Total_Quantity"
].idxmax()

best_quantity = category_sales[
    "Total_Quantity"
].max()

print(
    f"\nBest-selling category: "
    f"{best_category}"
)

print(
    f"Quantity sold: "
    f"{best_quantity}"
)


# %%
# ============================================================
# 8. CATEGORY REVENUE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    category_sales.index,
    category_sales["Total_Revenue"]
)

plt.title(
    "Revenue by Product Category"
)

plt.xlabel(
    "Product Category"
)

plt.ylabel(
    "Total Revenue"
)

plt.legend(
    ["Revenue"]
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
# 9. INTERPRETATION — CATEGORY GRAPH
# ============================================================

print("\nINTERPRETATION — CATEGORY REVENUE")

print(
    "The bar chart compares the total revenue generated "
    "by each product category."
)

print(
    f"The category with the highest quantity sold is "
    f"{best_category}."
)

print(
    "Differences between categories indicate which product "
    "groups contribute more strongly to overall sales."
)


# %%
# ============================================================
# 10. MONTHLY REVENUE TIME-SERIES PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue["Month"],
    monthly_revenue["Total Amount"],
    marker="o",
    label="Monthly Revenue"
)

plt.title(
    "Monthly Revenue Trend"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Revenue"
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
# 11. INTERPRETATION — TIME SERIES
# ============================================================

highest_month = monthly_revenue.loc[
    monthly_revenue["Total Amount"].idxmax()
]

lowest_month = monthly_revenue.loc[
    monthly_revenue["Total Amount"].idxmin()
]

print("\nINTERPRETATION — MONTHLY REVENUE")

print(
    "The time-series plot shows how total revenue changed "
    "over the observed months."
)

print(
    f"The highest monthly revenue occurred in "
    f"{highest_month['Month'].strftime('%B %Y')}."
)

print(
    f"The lowest monthly revenue occurred in "
    f"{lowest_month['Month'].strftime('%B %Y')}."
)

print(
    "The variation between months indicates fluctuations "
    "in retail sales performance."
)


# %%
# ============================================================
# 12. IQR OUTLIER DETECTION
# ============================================================

# Calculate Q1 and Q3
Q1 = df["Total Amount"].quantile(0.25)

Q3 = df["Total Amount"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper limits
lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


# Identify outliers
outliers = df[
    (df["Total Amount"] < lower_bound)
    |
    (df["Total Amount"] > upper_bound)
]


print("\n" + "=" * 60)
print("IQR OUTLIER DETECTION")
print("=" * 60)

print(
    f"\nQ1: {Q1:.2f}"
)

print(
    f"Q3: {Q3:.2f}"
)

print(
    f"IQR: {IQR:.2f}"
)

print(
    f"Lower Bound: {lower_bound:.2f}"
)

print(
    f"Upper Bound: {upper_bound:.2f}"
)

print(
    f"Number of Outliers: {len(outliers)}"
)

print("\nOutlier Records:")
print(
    outliers[
        [
            "Date",
            "Product Category",
            "Quantity",
            "Price per Unit",
            "Total Amount"
        ]
    ].head(20)
)


# %%
# ============================================================
# 13. OUTLIER BOXPLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["Total Amount"],
    vert=True
)

plt.title(
    "Boxplot of Transaction Revenue"
)

plt.xlabel(
    "Revenue"
)

plt.ylabel(
    "Total Amount"
)

plt.legend(
    ["Transaction Revenue"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# %%
# ============================================================
# 14. INTERPRETATION — OUTLIERS
# ============================================================

print("\nINTERPRETATION — IQR OUTLIER ANALYSIS")

print(
    f"The IQR method identified {len(outliers)} "
    "potential outlier transactions."
)

print(
    f"Transactions below {lower_bound:.2f} or above "
    f"{upper_bound:.2f} were classified as outliers."
)

print(
    "These transactions may represent unusually large "
    "or small sales compared with the majority of records."
)


# %%
# ============================================================
# 15. PREPARE MONTHLY DATA FOR FORECASTING
# ============================================================

# Create sequential time index
monthly_revenue["Time_Index"] = np.arange(
    len(monthly_revenue)
)

X = monthly_revenue[
    ["Time_Index"]
]

y = monthly_revenue[
    "Total Amount"
]


# %%
# ============================================================
# 16. BUILD LINEAR REGRESSION FORECAST MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X,
    y
)

# Historical predictions
monthly_revenue["Predicted Revenue"] = (
    model.predict(X)
)


# %%
# ============================================================
# 17. REGRESSION MODEL RESULTS
# ============================================================

coefficient = model.coef_[0]

intercept = model.intercept_

r2 = r2_score(
    y,
    monthly_revenue["Predicted Revenue"]
)

mae = mean_absolute_error(
    y,
    monthly_revenue["Predicted Revenue"]
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        monthly_revenue["Predicted Revenue"]
    )
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION FORECAST MODEL")
print("=" * 60)

print(
    f"\nIntercept: {intercept:.2f}"
)

print(
    f"Monthly trend coefficient: "
    f"{coefficient:.2f}"
)

print(
    f"\nR² Score: {r2:.2f}"
)

print(
    f"MAE: {mae:.2f}"
)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    "\nRegression Equation:"
)

print(
    f"Revenue = {intercept:.2f} + "
    f"({coefficient:.2f} × Time Index)"
)


# %%
# ============================================================
# 18. FUTURE REVENUE FORECAST
# ============================================================

# Forecast next 6 months
forecast_months = 6

last_month = monthly_revenue[
    "Month"
].max()

future_dates = pd.date_range(
    start=last_month + pd.DateOffset(months=1),
    periods=forecast_months,
    freq="MS"
)

future_time_index = np.arange(
    len(monthly_revenue),
    len(monthly_revenue) + forecast_months
)

future_X = pd.DataFrame({
    "Time_Index": future_time_index
})

future_revenue = model.predict(
    future_X
)


forecast_df = pd.DataFrame({
    "Month": future_dates,
    "Forecasted Revenue": future_revenue
})


print("\n" + "=" * 60)
print("FUTURE REVENUE FORECAST")
print("=" * 60)

print(
    forecast_df.round(2)
)


# %%
# ============================================================
# 19. HISTORICAL + FUTURE REVENUE GRAPH
# ============================================================

plt.figure(figsize=(13, 7))

# Historical revenue
plt.plot(
    monthly_revenue["Month"],
    monthly_revenue["Total Amount"],
    marker="o",
    label="Historical Revenue"
)

# Regression trend
plt.plot(
    monthly_revenue["Month"],
    monthly_revenue["Predicted Revenue"],
    linestyle="--",
    label="Regression Trend"
)

# Future forecast
plt.plot(
    forecast_df["Month"],
    forecast_df["Forecasted Revenue"],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

plt.title(
    "Historical and Forecasted Monthly Revenue"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Revenue"
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
# 20. INTERPRETATION — FORECAST
# ============================================================

print("\nINTERPRETATION — REVENUE FORECAST")

print(
    "The graph shows historical monthly revenue together "
    "with the fitted linear regression trend and future forecast."
)

if coefficient > 0:
    print(
        "The positive regression coefficient indicates an "
        "overall increasing revenue trend."
    )
elif coefficient < 0:
    print(
        "The negative regression coefficient indicates an "
        "overall decreasing revenue trend."
    )
else:
    print(
        "The regression coefficient is close to zero, "
        "indicating a relatively stable trend."
    )

print(
    f"The model produced an R² score of {r2:.2f} on the "
    "historical monthly data."
)

print(
    f"The model forecasts revenue for the next "
    f"{forecast_months} months."
)


# %%
# ============================================================
# 21. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print(
    f"\nThe dataset contains {len(df)} valid retail "
    "transactions after cleaning."
)

print(
    f"The best-selling category by quantity is "
    f"{best_category}, with {best_quantity} units sold."
)

print(
    f"The IQR method identified {len(outliers)} "
    "potential revenue outliers."
)

print(
    f"The linear regression model achieved an R² score "
    f"of {r2:.2f}."
)

print(
    "The model was then used to forecast revenue for "
    "the next six months."
)

print(
    "The forecasting visualization compares historical "
    "revenue, the regression trend and future predicted revenue."
)

print(
    "\nSales forecasting analysis completed successfully."
)