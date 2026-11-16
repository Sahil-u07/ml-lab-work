import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
dataset_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
csv_file = [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(dataset_path, csv_file))
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
X_simple = df[['area']]
y = df['price']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
simple_lr = LinearRegression()
simple_lr.fit(X_train_s, y_train_s)
y_pred_s = simple_lr.predict(X_test_s)
print("--- SIMPLE LINEAR REGRESSION RESULTS ---")
print(f"Intercept: {simple_lr.intercept_}")
print(f"Coefficient (Area): {simple_lr.coef_[0]}")
print(f"MSE: {mean_squared_error(y_test_s, y_pred_s):.2f}")
print(f"R2 Score: {r2_score(y_test_s, y_pred_s):.4f}")
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.title('Graph 1: Simple Linear Regression (Price vs Area)')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.show()
X_multi = df.drop(columns=['price'])
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
multi_lr = LinearRegression()
multi_lr.fit(X_train_m, y_train_m)
y_pred_m = multi_lr.predict(X_test_m)
print("\n--- MULTIPLE LINEAR REGRESSION RESULTS ---")
print(f"Intercept: {multi_lr.intercept_}")
print(f"MSE: {mean_squared_error(y_test_m, y_pred_m):.2f}")
print(f"R2 Score: {r2_score(y_test_m, y_pred_m):.4f}")
coef_series = pd.Series(multi_lr.coef_, index=X_multi.columns).sort_values()
plt.figure(figsize=(10, 5))
coef_series.plot(kind='barh', color='teal')
plt.title('Graph 2: Multiple Linear Regression Coefficients')
plt.xlabel('Coefficient Magnitude')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(y_test_m, y_pred_m, color='green', alpha=0.6)
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'r--', lw=2)
plt.title('Graph 3: Actual vs Predicted Prices (Multiple Regression)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
