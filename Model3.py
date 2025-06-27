import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor  # KNN Regressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_curve, auc

# Load data
file_path = 'cleaned_data.csv'
data = pd.read_csv(file_path)

# Define Features and Target
X = data[['Year', 'Month', 'Day', 'Profit', 'Price (Before GST)', 'Discount (%)', 'Inventory (Stock)']]
y = data['Total Quantity Sold']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=5),  # KNN Regressor
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42)  # Keep for comparison
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred) * 100

    # Binary conversion for classification metrics
    y_test_bin = np.where(y_test > y_test.median(), 1, 0)
    y_pred_bin = np.where(y_pred > np.median(y_pred), 1, 0)
    accuracy = accuracy_score(y_test_bin, y_pred_bin) * 100
    precision = precision_score(y_test_bin, y_pred_bin) * 100
    recall = recall_score(y_test_bin, y_pred_bin) * 100

    results[name] = {
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

# Save best model (Random Forest in this case)
joblib.dump(results['Random Forest']['model'], 'best_sales_model_rf.pkl')

# Save model comparison results to CSV
model_comparison_df = pd.DataFrame({
    'Model': results.keys(),
    'RMSE': [metrics['rmse'] for metrics in results.values()],
    'R2 Score (%)': [metrics['r2'] for metrics in results.values()],
    'Accuracy (%)': [metrics['accuracy'] for metrics in results.values()],
    'Precision (%)': [metrics['precision'] for metrics in results.values()],
    'Recall (%)': [metrics['recall'] for metrics in results.values()]
})
model_comparison_df.to_csv('model_comparison_results.csv', index=False)

# Print and compare results
print("\n\033[1mModel Comparison Results:\033[0m")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R2 Score: {metrics['r2']:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall: {metrics['recall']:.2f}%")

# Visualization: Bar plots of metrics
metric_names = ['rmse', 'r2', 'accuracy', 'precision', 'recall']

for metric in metric_names:
    plt.figure(figsize=(8, 4))
    plt.title(f"Comparison of {metric.upper()} across Models")
    sns.barplot(x=list(results.keys()), y=[metrics[metric] for metrics in results.values()])
    plt.ylabel(metric.upper())
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# Prediction Function
model = results['Random Forest']['model']

def predict_sales_for_year_month(year, month):
    data_filtered = data[(data['Year'] == year) & (data['Month'] == month)]
    if len(data_filtered) == 0:
        print(f"No data available for {month}/{year}. Please check the input data.")
        return

    X_filtered = data_filtered[['Year', 'Month', 'Day', 'Profit', 'Price (Before GST)', 'Discount (%)', 'Inventory (Stock)']]
    y_filtered = data_filtered['Total Quantity Sold']

    y_pred_filtered = model.predict(X_filtered)
    total_predicted_sales_amount = np.sum(y_pred_filtered * data_filtered['Price (Before GST)'].values)
    actual_total_sales_amount = np.sum(y_filtered * data_filtered['Price (Before GST)'].values)

    print(f"\n=== Sales Prediction Summary for {month}/{year} ===")
    print(f"Predicted Total Quantity Sold: {np.sum(y_pred_filtered):.0f} units")
    print(f"Predicted Total Sales (\u20b9): \u20b9{total_predicted_sales_amount:,.2f}")
    print(f"Actual Total Quantity Sold: {np.sum(y_filtered):.0f} units")
    print(f"Actual Total Sales (\u20b9): \u20b9{actual_total_sales_amount:,.2f}")

    rmse = mean_squared_error(y_filtered, y_pred_filtered) ** 0.5
    r2 = r2_score(y_filtered, y_pred_filtered) * 100
    accuracy = accuracy_score(np.where(y_filtered > y_filtered.median(), 1, 0), np.where(y_pred_filtered > np.median(y_pred_filtered), 1, 0)) * 100
    precision = precision_score(np.where(y_filtered > y_filtered.median(), 1, 0), np.where(y_pred_filtered > np.median(y_pred_filtered), 1, 0)) * 100
    recall = recall_score(np.where(y_filtered > y_filtered.median(), 1, 0), np.where(y_pred_filtered > np.median(y_pred_filtered), 1, 0)) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"R\u00b2 Score: {r2:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")

    plot_graphs(y_filtered, y_pred_filtered)

    y_filtered_bin = np.where(y_filtered > y_filtered.median(), 1, 0)
    y_pred_filtered_bin = np.where(y_pred_filtered > np.median(y_pred_filtered), 1, 0)
    plot_roc_curve(y_filtered_bin, y_pred_filtered_bin)

# Plot Functions
def plot_graphs(actual_sales, predicted_sales):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_sales[:50], label='Actual Sales', marker='o')
    plt.plot(predicted_sales[:50], label='Predicted Sales', marker='x')
    plt.title('Actual vs Predicted Sales - Line Chart')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales Quantity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sns.scatterplot(x=actual_sales, y=predicted_sales)
    plt.title('Scatter Plot: Actual vs Predicted Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sns.boxplot(data=[actual_sales, predicted_sales], palette='pastel')
    plt.xticks([0, 1], ['Actual', 'Predicted'])
    plt.title('Box Plot: Actual vs Predicted Distribution')
    plt.tight_layout()
    plt.show()

# ROC Curve Plot
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    year = int(input("Enter the year to predict sales (e.g., 2023): "))
    month = int(input("Enter the month to predict sales (1-12): "))
    predict_sales_for_year_month(year, month)
