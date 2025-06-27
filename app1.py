import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_curve, auc

# Load Model and Data
def load_model():
    return joblib.load('best_sales_model_rf.pkl')

def load_data():
    return pd.read_csv('cleaned_data.csv')

def load_model_results():
    return pd.read_csv("model_comparison_results.csv")

# Prediction Function
def predict_sales(model, features):
    return model.predict(features)

# Performance Metrics
def evaluate_model(y_true, y_pred):
    return {
        'RMSE': round(mean_squared_error(y_true, y_pred) ** 0.5, 2),
        'R2 Score (%)': round(r2_score(y_true, y_pred) * 100, 2),
        'Accuracy (%)': round(accuracy_score(np.where(y_true > y_true.median(), 1, 0), np.where(y_pred > np.median(y_pred), 1, 0)) * 100, 2),
        'Precision (%)': round(precision_score(np.where(y_true > y_true.median(), 1, 0), np.where(y_pred > np.median(y_pred), 1, 0)) * 100, 2),
        'Recall (%)': round(recall_score(np.where(y_true > y_true.median(), 1, 0), np.where(y_pred > np.median(y_pred), 1, 0)) * 100, 2)
    }

# Model Performance Visualizations
def plot_performance_charts(y_true, y_pred, data, model):
    st.markdown("## 📈 Model Performance Insights")

    # Scatter Plot
    st.subheader("📌 Scatter Plot: Actual vs Predicted")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax1, color='teal')
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    ax1.set_xlabel('Actual Sales')
    ax1.set_ylabel('Predicted Sales')
    st.pyplot(fig1)

    # Box Plot
    st.subheader("📦 Box Plot: Distribution of Actual vs Predicted Sales")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=[y_true, y_pred], palette='pastel', ax=ax2)
    ax2.set_xticklabels(['Actual', 'Predicted'])
    st.pyplot(fig2)

    # ROC Curve
    st.subheader("🧪 ROC Curve")
    fig3, ax3 = plt.subplots()
    fpr, tpr, _ = roc_curve(np.where(y_true > y_true.median(), 1, 0), np.where(y_pred > np.median(y_pred), 1, 0))
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='green')
    ax3.plot([0, 1], [0, 1], linestyle='--', color='black')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend()
    st.pyplot(fig3)

    # Correlation Heatmap
    st.subheader("🧲 Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

    # Feature Importance
    st.subheader("⭐ Feature Importance")
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_names = data[['Year', 'Month', 'Day', 'Profit', 'Price (Before GST)', 'Discount (%)', 'Inventory (Stock)']].columns
        if len(feature_importances) == len(feature_names):
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False)
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_df, palette='Spectral', ax=ax5)
            st.pyplot(fig5)

# Model Comparison Visualizations
def plot_model_comparison():
    st.markdown("## 🤖 Model Comparison Visualizations")
    results_df = load_model_results()

    # Barplot for R2 Score
    st.subheader("R² Score Comparison")
    fig1, ax1 = plt.subplots()
    sns.barplot(x='R2 Score (%)', y='Model', data=results_df.sort_values(by='R2 Score (%)', ascending=False), palette='coolwarm', ax=ax1)
    ax1.set_xlabel("R2 Score (%)")
    st.pyplot(fig1)

    # Barplot for RMSE
    st.subheader("RMSE Comparison")
    fig2, ax2 = plt.subplots()
    sns.barplot(x='RMSE', y='Model', data=results_df.sort_values(by='RMSE'), palette='viridis', ax=ax2)
    ax2.set_xlabel("RMSE")
    st.pyplot(fig2)

    # Accuracy, Precision, Recall comparison
    st.subheader("Accuracy, Precision, Recall Comparison")
    melted_df = results_df.melt(id_vars='Model', value_vars=['Accuracy (%)', 'Precision (%)', 'Recall (%)'], var_name='Metric', value_name='Score')
    fig3, ax3 = plt.subplots()
    sns.barplot(x='Score', y='Model', hue='Metric', data=melted_df, palette='Set2', ax=ax3)
    st.pyplot(fig3)

# Data Visualizations
def plot_basic_charts(data):
    st.markdown("## 📊 Basic Data Visualizations")

    # Line Chart
    st.subheader("📈 Sales Trend (Line Chart)")
    trend = data.groupby(['Year', 'Month'])['Total Quantity Sold'].sum().reset_index()
    trend['Date'] = pd.to_datetime(trend[['Year', 'Month']].assign(Day=1))
    st.line_chart(trend.set_index('Date')['Total Quantity Sold'])

    # Bar Chart for Product Category
    st.subheader("📊 Sales by Product Category")
    category_sales = data.groupby('Category')['Total Quantity Sold'].sum()
    st.bar_chart(category_sales)

    # Bar Chart: Top 10 Products
    st.subheader("📊 Top 10 Selling Products")
    top_products = data.groupby('Product Name')['Total Quantity Sold'].sum().sort_values(ascending=False).head(10)
    fig6, ax6 = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax6)
    ax6.set_xlabel('Total Quantity Sold')
    ax6.set_ylabel('Product Name')
    st.pyplot(fig6)

    # Horizontal Bar Chart: Top 10 Customer Addresses
    st.subheader("🏡 Top 10 Customer Locations")
    top_addresses = data.groupby('Customer Address')['Total Quantity Sold'].sum().sort_values(ascending=False).head(10)
    fig7, ax7 = plt.subplots()
    sns.barplot(x=top_addresses.values, y=top_addresses.index, palette='pastel', ax=ax7)
    ax7.set_xlabel('Total Quantity Sold')
    ax7.set_ylabel('Customer Address')
    st.pyplot(fig7)

    # Cumulative Line Chart for Total Sales
    st.subheader("📊 Cumulative Sales Over Time")
    cumulative_sales = data.groupby(['Year', 'Month'])['Total Quantity Sold'].sum().cumsum().reset_index()
    cumulative_sales['Date'] = pd.to_datetime(cumulative_sales[['Year', 'Month']].assign(Day=1))
    fig8, ax8 = plt.subplots()
    sns.lineplot(x='Date', y='Total Quantity Sold', data=cumulative_sales, marker='o', color='blue', ax=ax8)
    ax8.set_ylabel('Cumulative Quantity Sold')
    st.pyplot(fig8)

    # Histogram
    st.subheader("📊 Histogram of Sales Quantity")
    fig9, ax9 = plt.subplots()
    sns.histplot(data['Total Quantity Sold'], bins=20, kde=True, color='orange', ax=ax9)
    st.pyplot(fig9)

# Streamlit App Setup
st.set_page_config(page_title="R K Traders Sales Prediction Dashboard", layout="wide")
st.title("📊 R K Traders Sales Prediction Website")
st.markdown("""
Welcome to the **R K Traders** AI-powered dashboard.
Select inputs to view predictions, total sales, and optionally performance metrics and visual insights.
""")

# Inputs
model = load_model()
data = load_data()

st.sidebar.header("🔵 User Input Panel")
year = st.sidebar.selectbox("Select Year", sorted(data['Year'].unique()))
month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

# Advanced Filters
st.sidebar.subheader("🔴 Advanced Input Options")
apply_advanced_filters = st.sidebar.checkbox("Apply Advanced Filters")

if apply_advanced_filters:
    min_sales = st.sidebar.number_input("Minimum Sales Quantity", min_value=0, step=1, value=0)
    max_sales = st.sidebar.number_input("Maximum Sales Quantity", min_value=0, step=1, value=10000)
    selected_product = st.sidebar.selectbox("Select Product Category", data['Category'].unique())

# Optional Checkboxes for Performance & Visualization
show_metrics = st.sidebar.checkbox("📈 Show Model Performance Metrics")
show_visuals = st.sidebar.checkbox("📊 Show Data Visualizations")
show_comparison = st.sidebar.checkbox("🤖 Show Model Comparison")

if st.sidebar.button("Predict Sales"):
    filtered_df = data[(data['Year'] == year) & (data['Month'] == month)]

    if apply_advanced_filters:
        filtered_df = filtered_df[(filtered_df['Total Quantity Sold'] >= min_sales) & (filtered_df['Total Quantity Sold'] <= max_sales)]
        filtered_df = filtered_df[filtered_df['Category'] == selected_product]

    if filtered_df.empty:
        st.warning(f"No records found for {month}/{year} with selected filters. Try different inputs.")
    else:
        X_filtered = filtered_df[['Year', 'Month', 'Day', 'Profit', 'Price (Before GST)', 'Discount (%)', 'Inventory (Stock)']]
        y_filtered = filtered_df['Total Quantity Sold']

        y_pred_filtered = predict_sales(model, X_filtered)
        total_predicted_sales_amount = np.sum(y_pred_filtered * filtered_df['Price (Before GST)'].values)
        actual_total_sales_amount = np.sum(y_filtered * filtered_df['Price (Before GST)'].values)

        st.subheader(f"🔵 Sales Prediction Summary for {month}/{year}")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Total Quantity Sold", f"{np.sum(y_pred_filtered):.0f} units")
        col1.metric("Predicted Total Sales (₹)", f"₹{total_predicted_sales_amount:,.2f}")
        col2.metric("Actual Total Quantity Sold", f"{np.sum(y_filtered):.0f} units")
        col2.metric("Actual Total Sales (₹)", f"₹{actual_total_sales_amount:,.2f}")

        if show_metrics:
            st.subheader("📈 Performance Metrics")
            metrics = evaluate_model(y_filtered, y_pred_filtered)
            for k, v in metrics.items():
                st.write(f"**{k}**: {v}")
            plot_performance_charts(y_filtered, y_pred_filtered, data, model)

        if show_visuals:
            plot_basic_charts(data)

        if show_comparison:
            plot_model_comparison()
