# Machine Learning Lab

Sahil Lenka, Reg. No. 2427010391, Section C
B.Tech CSE (AIML), Manipal University Jaipur
Course: AIM 3130 Machine Learning Lab

| Lab | Topic | Data |
|-----|-------|------|
| 1 | Logistic regression with SGD | Kaggle `yasserh/titanic-dataset` |
| 2 | Feature scaling and normalization | Kaggle `yasserh/titanic-dataset` |
| 3 | Exploratory data analysis | Kaggle `uciml/iris` |
| 4 | Linear regression | Kaggle `yasserh/housing-prices-dataset` |
| 5 | Logistic regression and ROC | scikit-learn breast cancer |
| 6 | Naive Bayes classification | Kaggle stroke prediction CSV |
| 7 | Decision trees and random forests | stroke CSV, UCI concrete strength |
| 8 | K-Means and DBSCAN clustering | stroke CSV |

## Running

```
pip install -r requirements.txt
python Lab1/experiment1.py
```

Labs 1 to 4 download their data with `kagglehub`, which needs Kaggle credentials (`~/.kaggle/kaggle.json`). Lab 5 needs no download.

Labs 6 to 8 read these files from the folder you run them in:

- `healthcare-dataset-stroke-data.csv` (Kaggle, "Stroke Prediction Dataset", 5110 rows)
- `Concrete_Data.xls` (UCI, "Concrete Compressive Strength", 1030 rows), Lab 7 only

```
cd Lab6 && python experiment6.py
```

Labs 7 and 8 open matplotlib windows, so run them somewhere with a display.

## Results (Labs 6 to 8)

Lab 6, test accuracy on the stroke data: Gaussian NB 0.868, Multinomial NB 0.910, Bernoulli NB 0.941, Categorical NB 0.946, mixed Gaussian + Categorical NB 0.897.

Lab 7, concrete strength (regression), test RMSE and R2: unpruned tree 6.905 / 0.815, pruned tree 6.966 / 0.812, random forest 5.568 / 0.880.

Lab 8, K-Means picks K = 4 by silhouette (0.227). The best DBSCAN setting is eps = 2.5, min_samples = 5, which gives 4 clusters and a silhouette of 0.274.

The stroke data is heavily imbalanced (about 5% positive). Accuracy looks high for that reason, and the Lab 7 random forest classifier reaches 0.948 accuracy with an F1 of 0 on the stroke class.

Tested with numpy 2.5, pandas 3.0, matplotlib 3.11, scikit-learn 1.9. Labs 6 to 8 were run end to end. Labs 1 to 5 were not re-run for this README because they need the Kaggle downloads.
