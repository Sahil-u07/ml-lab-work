import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("--- 1. LOADING DATASET FROM KAGGLE ---")
# Downloads the public Titanic dataset via kagglehub
dataset_path = kagglehub.dataset_download("yasserh/titanic-dataset")
csv_file = [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0]
full_csv_path = os.path.join(dataset_path, csv_file)
df = pd.read_csv(full_csv_path)
print(f"Dataset successfully loaded from: {full_csv_path}\n")
print("First 5 rows of the raw dataset:")
print(df.head())
print("\nDataset Summary & Structure:")
print(df.info())
print("\n--- 2. NUMPY BASIC ARRAY OPERATIONS ---")
temp_age = df['Age'].fillna(df['Age'].median()).values
temp_fare = df['Fare'].fillna(df['Fare'].median()).values
data_matrix = np.column_stack((temp_age, temp_fare))
print(f"NumPy Array Shape: {data_matrix.shape}")
print(f"Mean (Age, Fare): {np.mean(data_matrix, axis=0)}")
print(f"Standard Deviation (Age, Fare): {np.std(data_matrix, axis=0)}")
print(f"Max Values (Age, Fare): {np.max(data_matrix, axis=0)}")
print(f"Min Values (Age, Fare): {np.min(data_matrix, axis=0)}")
subset_matrix = data_matrix[:10]
reshaped_matrix = subset_matrix.reshape(5, 4)
print("\nReshaping Demonstration (First 10 rows from 10x2 to 5x4):")
print(reshaped_matrix)
print("\n--- 3. FEATURE ENGINEERING ---")
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
                'Sex_male', 'Embarked_Q', 'Embarked_S']
X = df[feature_cols]
y = df['Survived']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("Head of Scaled Feature Matrix:")
print(X_scaled_df.head())
print("\n--- 4. SPLITTING DATASET INTO TRAIN & TEST SETS ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
print(f"X_test Shape:  {X_test.shape}  | y_test Shape:  {y_test.shape}")
print("\n--- 5. GENERATING GRAPHS ---")
plt.figure(figsize=(9, 7))
corr_matrix = df[feature_cols + ['Survived']].corr()
cax = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(cax)
cols = corr_matrix.columns
plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        val = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha='center', va='center',
                 color='white' if abs(val) > 0.5 else 'black', fontsize=8)
plt.title('Graph 1: Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42)
epochs = 50
train_loss = []
for epoch in range(epochs):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    train_loss.append(log_loss(y_train, y_prob))
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_loss, marker='o', color='#d95f02', linewidth=2, label='Training Loss')
plt.title('Graph 2: Model Training Loss Curve over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did Not Survive', 'Survived'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Graph 3: Confusion Matrix on Test Set', fontsize=14, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.show()
print("\nExperiment 1 successfully completed!")
