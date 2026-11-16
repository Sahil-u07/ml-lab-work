import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset_path = kagglehub.dataset_download("uciml/iris")
csv_file = [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(dataset_path, csv_file))
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
print("--- DESCRIPTIVE STATISTICS ---")
print(df.describe())
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    plt.hist(df[col], bins=15, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'], c='purple', alpha=0.7)
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
df[numeric_cols].boxplot()
plt.title('Box Plots for Numerical Features')
plt.ylabel('Values (cm)')
plt.show()
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_cols].corr()
cax = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(cax)
cols = corr_matrix.columns
plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        val = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha='center', va='center',
                 color='white' if abs(val) > 0.5 else 'black')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
n_feat = len(numeric_cols)
fig, axes = plt.subplots(n_feat, n_feat, figsize=(10, 10))
for i in range(n_feat):
    for j in range(n_feat):
        if i == j:
            axes[i, j].hist(df[numeric_cols[i]], color='gray')
        else:
            axes[i, j].scatter(df[numeric_cols[j]], df[numeric_cols[i]], s=10, alpha=0.6)
        if i == n_feat - 1:
            axes[i, j].set_xlabel(numeric_cols[j], fontsize=8)
        if j == 0:
            axes[i, j].set_ylabel(numeric_cols[i], fontsize=8)
plt.suptitle('Pair Plot Matrix', y=1.02)
plt.tight_layout()
plt.show()
