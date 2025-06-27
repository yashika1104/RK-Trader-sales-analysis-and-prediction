import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'cleaned_data.csv'
data = pd.read_csv(file_path)

# Basic Descriptive Statistics
print("Data Size:\n",data.shape)
print("Data Columns:\n",data.columns)
print("Data Information:\n",data.info)
print("Summary Statistics:\n", data.describe())
print("Data Types:\n", data.dtypes)
print("Unique Values per Column:\n", data.nunique())

# Ensure all numerical columns are correctly formatted
numerical_cols = ['Total Quantity Sold', 'Profit', 'Price (Before GST)', 'GST Amount', 'Discount (%)', 'Total Price (With GST)']
data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Remove any remaining NaN values after conversion
data.dropna(inplace=True)

# Display unique values in categorical columns
categorical_cols = ['Product Name', 'Category', 'Customer Name']
for col in categorical_cols:
    print(f"Unique values in {col}:", data[col].nunique())

# Correlation Heatmap (Fix: Use only numeric columns)
numeric_data = data.select_dtypes(include=[np.number])  
corr = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Sales Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Transaction Date', y='Total Quantity Sold', hue='Category', marker='o')
plt.title('Sales Trend Over Time')
plt.xticks(rotation=45)
plt.show()

# Sales vs Profit Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Total Quantity Sold', y='Profit', hue='Category')
plt.title('Sales vs Profit Analysis')
plt.show()

# Distribution of Numerical Features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Boxplot for Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Total Quantity Sold', data=data)
plt.title('Sales Distribution by Product Category')
plt.xticks(rotation=45)
plt.show()

# Monthly Sales Analysis
monthly_sales = data.groupby(['Year', 'Month'])['Total Quantity Sold'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_sales, x='Month', y='Total Quantity Sold', hue='Year')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.show()

# Customer Spending Analysis
data['Customer Spend'] = data['Total Quantity Sold'] * data['Price (Before GST)']
plt.figure(figsize=(10, 6))
sns.histplot(data['Customer Spend'], bins=30, kde=True)
plt.title('Customer Spending Distribution')
plt.show()

# Top Selling Products
top_products = data.groupby('Product Name')['Total Quantity Sold'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Best Selling Products')
plt.xticks(rotation=45)
plt.show()

# Discount vs Sales Analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Discount (%)', y='Total Quantity Sold')
plt.title('Impact of Discount on Sales')
plt.show()