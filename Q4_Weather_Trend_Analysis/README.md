# Q4 – Weather Trend Analysis

## Objective

Analyze weather data to identify temperature and rainfall trends, calculate monthly averages, detect unusually hot days using the Z-score method, and build a Linear Regression model for temperature prediction.

## Dataset

- **Source:** Kaggle
- **Dataset:** Weather Data
- **File:** weather_data.csv
- **Columns:** Location, Date_Time, Temperature_C, Humidity_pct, Precipitation_mm, Wind_Speed_kmh

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Analysis Performed

1. Loaded and inspected the weather dataset.
2. Converted the Date_Time column to datetime format.
3. Checked for missing and duplicate values.
4. Cleaned the dataset.
5. Calculated monthly average temperature and rainfall.
6. Identified unusually hot days using the Z-score method.
7. Created temperature trend visualizations.
8. Created rainfall visualizations.
9. Built a Linear Regression model.
10. Evaluated the model using R² and Mean Squared Error (MSE).

## Visualizations

The analysis includes:

- Monthly average temperature
- Monthly rainfall
- Temperature distribution
- Unusually hot days
- Actual vs predicted temperature

## Machine Learning

Linear Regression is used to predict temperature based on selected weather-related features.

The model is evaluated using:

- R² Score
- Mean Squared Error (MSE)

## Conclusion

The analysis identifies temperature and rainfall trends in the weather dataset and highlights unusually hot observations using statistical analysis. The Linear Regression model provides a basic approach for predicting temperature from weather-related variables.