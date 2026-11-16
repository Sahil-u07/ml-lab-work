# Machine Learning Lab

Sahil Lenka, Reg. No. 2427010391, Section C
B.Tech CSE (AIML), Manipal University Jaipur

| Lab | Topic | Data |
|-----|-------|------|
| 1 | Logistic regression with SGD | Kaggle `yasserh/titanic-dataset` |
| 2 | Feature scaling and normalization | Kaggle `yasserh/titanic-dataset` |
| 3 | Exploratory data analysis | Kaggle `uciml/iris` |
| 4 | Linear regression | Kaggle `yasserh/housing-prices-dataset` |
| 5 | Logistic regression and ROC | scikit-learn breast cancer |
| 6 | Naive Bayes classification | `healthcare-dataset-stroke-data.csv` |
| 7 | Decision trees and random forests | stroke CSV, `Concrete_Data.xls` (UCI) |
| 8 | K-Means and DBSCAN clustering | stroke CSV |

Each folder has `experimentN.py`. Labs 1 to 4 download their data with `kagglehub`. Labs 6 to 8 read the CSV/XLS from the working directory; the files are not included.

```
pip install numpy pandas matplotlib scikit-learn kagglehub xlrd
python Lab1/experiment1.py
```
