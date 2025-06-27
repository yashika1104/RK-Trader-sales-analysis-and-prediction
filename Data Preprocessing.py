import pandas as pd
import numpy as np

# Load Data
file_path = 'R.K_furniture_store_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Check column names
print("Columns in dataset:", data.columns)

# Rename columns if necessary
data.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure 'Transaction Date' column exists
if 'Transaction Date' not in data.columns:
    raise KeyError("The column 'Transaction Date' is missing from the dataset. Please check the column names.")

# Basic Information and Cleaning
print("Initial data shape:", data.shape)
print("Data Info:\n", data.info())
print("Missing Values:\n", data.isnull().sum())

# Remove Duplicates
data.drop_duplicates(inplace=True)

# Fill Missing Values
data = data.ffill().bfill()

# Convert Data Types
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'], errors='coerce')
data.dropna(subset=['Transaction Date'], inplace=True)  # Remove rows where Date conversion failed
data['Total Quantity Sold'] = pd.to_numeric(data['Total Quantity Sold'], errors='coerce')
data['Profit'] = pd.to_numeric(data['Profit'], errors='coerce')
data['Price (Before GST)'] = pd.to_numeric(data['Price (Before GST)'], errors='coerce')
data.dropna(subset=['Total Quantity Sold', 'Profit', 'Price (Before GST)'], inplace=True)

# Remove Outliers using IQR
Q1 = data['Total Quantity Sold'].quantile(0.25)
Q3 = data['Total Quantity Sold'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Total Quantity Sold'] >= (Q1 - 1.5 * IQR)) & (data['Total Quantity Sold'] <= (Q3 + 1.5 * IQR))]

# Feature Engineering
data['Year'] = data['Transaction Date'].dt.year
data['Month'] = data['Transaction Date'].dt.month
data['Day'] = data['Transaction Date'].dt.day

data.to_csv('cleaned_data.csv', index=False)
print("Data Preprocessing Completed. Final shape:", data.shape)