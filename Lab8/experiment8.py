import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
 
# 1. Load & preprocess
df = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns=["id", "stroke"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
 
# 2. PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
# 3. K-Means: Elbow Method
wcss = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
 
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, wcss, marker="o")
ax[0].set_xlabel("K"); ax[0].set_ylabel("WCSS"); ax[0].set_title("Elbow Method")
ax[1].plot(K_range, sil_scores, marker="o", color="green")
ax[1].set_xlabel("K"); ax[1].set_ylabel("Silhouette Score"); ax[1].set_title("Silhouette Analysis")
plt.tight_layout(); plt.show()
 
best_k = K_range[np.argmax(sil_scores)]
print("Optimal K (by silhouette):", best_k)
 
# 4. Final K-Means model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = kmeans_final.fit_predict(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="tab10", s=15)
plt.title(f"K-Means Clustering (K={best_k})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
 
# 5. DBSCAN: k-distance plot to help choose eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel("Points sorted by distance"); plt.ylabel("5th NN distance")
plt.title("k-distance Plot (for choosing eps)")
plt.show()
 
# 6. DBSCAN tuning
best_combo, best_score = None, -1
for eps in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for min_samples in [5, 10, 15]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score, best_combo = score, (eps, min_samples, n_clusters)
print("Best DBSCAN (eps, min_samples, n_clusters):", best_combo, "Silhouette:", best_score)
 
# 7. Final DBSCAN model
eps_final, min_samples_final, _ = best_combo
dbscan_final = DBSCAN(eps=eps_final, min_samples=min_samples_final)
db_labels = dbscan_final.fit_predict(X_scaled)
n_noise = np.sum(db_labels == -1)
print("Noise points:", n_noise)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="tab10", s=15)
plt.title("DBSCAN Clustering")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
 
# 8. Comparison
print("K-Means Silhouette:", silhouette_score(X_scaled, km_labels))
print("DBSCAN Silhouette:", best_score)