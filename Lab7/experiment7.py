import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor, plot_tree)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_squared_error, r2_score)
 
np.random.seed(42)
# Part A: Classification (Stroke)
# 1. Load & preprocess
df = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns=["id"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop(columns=["stroke"])
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# validation split (taken from the training data) used for choosing the pruning level
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
 
# 2. Baseline Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Baseline DT Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
 
# 3. Splitting criterion comparison
for crit in ["gini", "entropy"]:
    m = DecisionTreeClassifier(criterion=crit, random_state=42).fit(X_train, y_train)
    print(f"Criterion={crit}: F1={f1_score(y_test, m.predict(X_test)):.4f}")
 
# 4. Effect of max_depth (overfitting analysis)
depths = range(1, 21)
train_acc, test_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train)))
    test_acc.append(accuracy_score(y_test, m.predict(X_test)))
plt.plot(depths, train_acc, label="Train")
plt.plot(depths, test_acc, label="Test")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Decision Tree: Depth vs Accuracy")
plt.show()
 
# 4b. Effect of min_samples_split
print("\nmin_samples_split sweep (Decision Tree classifier):")
for mss in [2, 5, 10, 20, 50, 100]:
    m = DecisionTreeClassifier(min_samples_split=mss, random_state=42).fit(X_train, y_train)
    print(f"  min_samples_split={mss:<4} train_acc={accuracy_score(y_train, m.predict(X_train)):.4f} "
          f"test_acc={accuracy_score(y_test, m.predict(X_test)):.4f} "
          f"test_F1={f1_score(y_test, m.predict(X_test)):.4f} leaves={m.get_n_leaves()}")
 
# 5. Cost-complexity pruning (alpha chosen on the validation set, not the test set)
path = dt.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.98, 25)))
val_scores = []
for alpha in ccp_alphas:
    m = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_tr, y_tr)
    val_scores.append(f1_score(y_val, m.predict(X_val)))
best_alpha = ccp_alphas[int(np.argmax(val_scores))]
pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha).fit(X_train, y_train)
print("\nBest ccp_alpha (validation):", best_alpha, " Val F1:", max(val_scores))
print("Leaves: unpruned =", dt.get_n_leaves(), " pruned =", pruned_tree.get_n_leaves())
 
plt.figure(figsize=(16, 7))
plot_tree(pruned_tree, max_depth=3, feature_names=X.columns, class_names=["no stroke", "stroke"],
          filled=True, fontsize=7)
plt.title("Pruned Decision Tree (top 3 levels)")
plt.show()
 
# 6. Random Forest with hyperparameter tuning
rf_params = {"n_estimators": [100, 200], "max_depth": [5, 10, None],
             "max_features": ["sqrt", "log2"]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)
 
# 7. Compare Decision Tree vs Random Forest
start = time.time(); dt_pruned_pred = pruned_tree.predict(X_test); dt_time = time.time() - start
start = time.time(); rf_pred = best_rf.predict(X_test); rf_time = time.time() - start
print(f"Pruned DT: Acc={accuracy_score(y_test, dt_pruned_pred):.4f}, "
      f"F1={f1_score(y_test, dt_pruned_pred):.4f}, Time={dt_time:.4f}s")
print(f"Random Forest: Acc={accuracy_score(y_test, rf_pred):.4f}, "
      f"F1={f1_score(y_test, rf_pred):.4f}, Time={rf_time:.4f}s")
 
# 8. Feature importance comparison
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "DecisionTree": pruned_tree.feature_importances_,
    "RandomForest": best_rf.feature_importances_
}).sort_values("RandomForest", ascending=False)
print(importance_df)
importance_df.set_index("Feature").plot(kind="bar", figsize=(10, 5))
plt.title("Feature Importance: DT vs RF (classification)")
plt.tight_layout(); plt.show()
 
# 9. Robustness to noise
noise = np.random.normal(0, X_test.std() * 0.5, X_test.shape)
X_noisy = X_test + noise
print("DT accuracy on noisy data:", accuracy_score(y_test, pruned_tree.predict(X_noisy)))
print("RF accuracy on noisy data:", accuracy_score(y_test, best_rf.predict(X_noisy)))
 
# Part B: Regression (Concrete)
print("\nRegression: Concrete Compressive Strength")
dfc = pd.read_excel("Concrete_Data.xls")
dfc.columns = ["cement","slag","fly_ash","water","superplasticizer",
               "coarse_agg","fine_agg","age","strength"]
Xc = dfc.drop(columns=["strength"])
yc = dfc["strength"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
 
# 10. Baseline tree and splitting criteria
for crit in ["squared_error", "friedman_mse"]:
    m = DecisionTreeRegressor(criterion=crit, random_state=42).fit(Xc_train, yc_train)
    print(f"Criterion={crit}: test RMSE={rmse(yc_test, m.predict(Xc_test)):.3f}  R2={r2_score(yc_test, m.predict(Xc_test)):.3f}")
 
# 11. max_depth vs error
tr_err, te_err = [], []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xc_train, yc_train)
    tr_err.append(rmse(yc_train, m.predict(Xc_train)))
    te_err.append(rmse(yc_test, m.predict(Xc_test)))
plt.plot(depths, tr_err, label="Train RMSE")
plt.plot(depths, te_err, label="Test RMSE")
plt.xlabel("max_depth"); plt.ylabel("RMSE"); plt.legend()
plt.title("Decision Tree Regressor: Depth vs RMSE")
plt.show()
 
# 12. min_samples_split sweep
print("\nmin_samples_split sweep (Decision Tree regressor):")
for mss in [2, 5, 10, 20, 50]:
    m = DecisionTreeRegressor(min_samples_split=mss, random_state=42).fit(Xc_train, yc_train)
    print(f"  min_samples_split={mss:<3} train_RMSE={rmse(yc_train, m.predict(Xc_train)):.3f} "
          f"test_RMSE={rmse(yc_test, m.predict(Xc_test)):.3f} leaves={m.get_n_leaves()}")
 
# 13. Cost-complexity pruning chosen by 5-fold CV on the training set
dtr = DecisionTreeRegressor(random_state=42).fit(Xc_train, yc_train)
cand = np.unique(np.quantile(dtr.cost_complexity_pruning_path(Xc_train, yc_train).ccp_alphas,
                             np.linspace(0, 0.98, 20)))
cv_rmse = [-cross_val_score(DecisionTreeRegressor(random_state=42, ccp_alpha=a), Xc_train, yc_train,
                            cv=5, scoring="neg_root_mean_squared_error").mean() for a in cand]
best_a = cand[int(np.argmin(cv_rmse))]
pruned_reg = DecisionTreeRegressor(random_state=42, ccp_alpha=best_a).fit(Xc_train, yc_train)
print(f"\nBest ccp_alpha (CV): {best_a:.4f}; leaves unpruned={dtr.get_n_leaves()} pruned={pruned_reg.get_n_leaves()}")
 
# 14. Random Forest regressor with GridSearchCV
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                        {"n_estimators": [100, 200], "max_depth": [10, 20, None],
                         "max_features": [0.5, 1.0]},
                        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
rfr_grid.fit(Xc_train, yc_train)
best_rfr = rfr_grid.best_estimator_
print("Best RF regressor params:", rfr_grid.best_params_)
 
# 15. Compare DT vs RF (regression)
rows = []
for name, mdl in [("Unpruned DT", dtr), ("Pruned DT", pruned_reg), ("Random Forest", best_rfr)]:
    s = time.time(); p = mdl.predict(Xc_test); t_ = time.time() - s
    rows.append((name, rmse(yc_test, p), r2_score(yc_test, p), t_))
print(pd.DataFrame(rows, columns=["Model", "Test_RMSE", "Test_R2", "Predict_time(s)"]).round(4).to_string(index=False))
 
imp = pd.DataFrame({"Feature": Xc.columns, "DecisionTree": pruned_reg.feature_importances_,
                    "RandomForest": best_rfr.feature_importances_}).sort_values("RandomForest", ascending=False)
print(imp.round(4).to_string(index=False))
imp.set_index("Feature").plot(kind="bar", figsize=(10, 5))
plt.title("Feature Importance: DT vs RF (regression)")
plt.tight_layout(); plt.show()
 
noisy = Xc_test + np.random.normal(0, Xc_test.std() * 0.2, Xc_test.shape)
print("Noisy-data RMSE  Pruned DT:", round(rmse(yc_test, pruned_reg.predict(noisy)), 3),
      " Random Forest:", round(rmse(yc_test, best_rfr.predict(noisy)), 3))