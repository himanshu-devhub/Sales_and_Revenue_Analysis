# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodel.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = "Sales_Analysis\train.csv"
sales_data = pd.read_csv(file_path)

# Display the first few rows
print(sales_data.info())


# DATA CLEANING
# Handle missing values
sales_data = sales_data.dropna()

# Convert date column to datetime
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])

# Ensure numeric columns are in correct format
sales_data['Sales'] = pd.to_numeric(sales_data['Sales'], errors= 'coerce')
sales_data['Quantity'] = pd.to_numeric(sales_data['Quantity'], errors ='coerce')

# Drop duplicates
sales_data = sales_data.drop_duplicate()

print("Cleaned data overview")
print(sales_data.info())

# Exploratory Data Analysis (EDA)
# Summary statistics 
print(sales_data.describe())

#Total sales by product category 
category_sales = sales_data.groupby('Category')['Sales'].sum().sort_values(ascending= False)
print(category_sales)

# Plot sales trends over time 
sales_data['YearMonth'] = sales_data['Order Date'].dt.to_period('M')
monthly_sales = sales_data.groupby('YearMonth')['Sales'].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot(kind='line', title='Monthly Sales Trend', color='blue')
plt.ylabel('Total Sales')
plt.xlabel('Month')
plt.show()

# Heatmap for correlations
plt.figure(figsize=(8,6))
sns.heatmap(sales_data.corr(),annot = True, cmap = 'coolwarm')
plt.show()

# Sales Forecasting (ARIMA MODEL)
#Preparing time series data
sales_ts = monthly_sales.reset_index()
sales_ts['YearMonth'] = sales_ts['YearMonth'].astype(str)
sales_ts.set_index('YearMonth', inplace = True)

# ARIMA Model for Forecasting
model = ARIMA(sales_ts['Sales'], order = (1, 1, 1)) #Adjust order as needed
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)
print("Forecast for next 12 months: ")

# Plot actual vs forecaste sales
plt.figure(figsize=(12, 6))
print("Forecast for next 12 months: ")
print(forecast)

# Plot actual Vs forecasted sales 
plt.fgure(figsize=(12, 6))
plt.plot(sales_ts, label = 'Actual Sales', color = 'blue')
plt.plot(pd.Series(forecast, index=pd.period_range(start=sales_ts.index[-1],periods =12, freq='M')),
        label='Forecasted Sales', color = 'red')
plt.legend()
plt.title("Actual vs Forecasted Sales")
plt.show()

# Key Factors Influencing Sales
region_sales = sales_data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
print(region_sales)

# Analyze sales by product
product_sales = sales_data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
print(product_sales)

# Visualization of top products
product_sales.plot(kind='bar', title ='Top 10 Product by Sales', color ='orange',figsize=(10,6))
plt.ylabel('Total Sales')
plt.xlabel('Product')
plt.show()

# Save cleaned data and analysis results
sales_data.to_csv("cleaned_sales_data.csv", index=False)
monthly_sales.to_csv("monthly_sales_trend.csv")