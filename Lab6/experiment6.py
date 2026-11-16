import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
# 1. Load & preprocess
df = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns=["id"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop(columns=["stroke"])
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
# 2. Gaussian Naive Bayes (numerical features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Gaussian Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))
 
# 3. Multinomial Naive Bayes (requires non-negative features -> discretize)
discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
X_disc = pd.DataFrame(discretizer.fit_transform(X), columns=X.columns)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_disc, y, test_size=0.2, random_state=42, stratify=y)
mnb = MultinomialNB()
mnb.fit(Xd_train, yd_train)
y_pred_mnb = mnb.predict(Xd_test)
print("\nMultinomial Naive Bayes:")
print("Accuracy:", accuracy_score(yd_test, y_pred_mnb))
print(classification_report(yd_test, y_pred_mnb))
 
# 4. Bernoulli Naive Bayes (binarized features)
X_bin = (X > X.median()).astype(int)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_bin, y, test_size=0.2, random_state=42, stratify=y)
bnb = BernoulliNB()
bnb.fit(Xb_train, yb_train)
y_pred_bnb = bnb.predict(Xb_test)
print("\nBernoulli Naive Bayes:")
print("Accuracy:", accuracy_score(yb_test, y_pred_bnb))
print(classification_report(yb_test, y_pred_bnb))
 
# 5. Cross-validation comparison
for name, model, data in [("Gaussian", gnb, X), ("Multinomial", mnb, X_disc), ("Bernoulli", bnb, X_bin)]:
    scores = cross_val_score(model, data, y, cv=5, scoring="f1")
    print(f"{name} NB CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
 
# 5b. Categorical Naive Bayes on the categorical features, and a mixed model that
#     combines Gaussian NB (numerical features) with Categorical NB (categorical features)
from sklearn.naive_bayes import CategoricalNB
cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type",
            "Residence_type", "smoking_status"]
num_cols = ["age", "avg_glucose_level", "bmi"]
n_cat = (X[cat_cols].max().values + 1)
cnb = CategoricalNB(min_categories=n_cat).fit(X_train[cat_cols], y_train)
y_pred_cnb = cnb.predict(X_test[cat_cols])
print("\nCategorical Naive Bayes (categorical features only):")
print("Accuracy:", accuracy_score(y_test, y_pred_cnb))
print(classification_report(y_test, y_pred_cnb))
 
gnb_num = GaussianNB().fit(X_train[num_cols], y_train)
log_prior = cnb.class_log_prior_
log_post = (gnb_num.predict_log_proba(X_test[num_cols])
            + cnb.predict_log_proba(X_test[cat_cols]) - log_prior)
y_pred_mix = log_post.argmax(axis=1)
print("Mixed NB (Gaussian for numeric + Categorical for categorical):")
print("Accuracy:", accuracy_score(y_test, y_pred_mix))
print(confusion_matrix(y_test, y_pred_mix))
print(classification_report(y_test, y_pred_mix))
 
# 6. Feature correlation (to discuss independence assumption)
print("\nFeature correlation matrix:")
print(X.corr().round(2))




