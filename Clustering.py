import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# import the column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label"
]
from sklearn.model_selection import train_test_split

# Load full dataset first
full_data = pd.read_csv('kddcup.data_10_percent', names=column_names)

# Stratified sampling based on label to preserve class proportions
data, _ = train_test_split(
    full_data,
    stratify=full_data['label'],
    test_size=0.85,
    random_state=42
)

print(data['label'].value_counts())
print(data.info())
print('Statistics Summary:', data.describe())
print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]}columns")
print('Labels:', data.iloc[:, 41].value_counts())
data.to_csv('NewDataset.csv', index=False)

# Check if there is any missing values
print(pd.DataFrame({'percent_missing': data.isnull().sum() * 100 / len(data)}))

# Prepare the dataset for processing

# Drop the last column which is the label
data.drop(columns=["label"], inplace=True)

# Removing features with unique values
drop_columns = [col for col in data.columns if data[col].nunique() == 1]

# Print the number and names of dropped columns
num_dropped = len(drop_columns)
dropped_columns_list = drop_columns
print("Number of columns with unique values to be dropped:", num_dropped)
print("Columns names with unique values to be dropped:", dropped_columns_list)
# Drop the columns
data.drop(columns=drop_columns, inplace=True)

# Group low-frequency categories before One-Hot Encoding
def group_rare_categories(df, column, threshold=0.02):
    value_counts = df[column].value_counts(normalize=True)
    common_values = value_counts[value_counts >= threshold].index
    df[column] = df[column].apply(lambda x: x if x in common_values else 'other')
    return df

# Apply to each categorical column
for col in ['protocol_type', 'service', 'flag']:
    data = group_rare_categories(data, col, threshold=0.01)

for col in ['protocol_type', 'service', 'flag']:
    print(f"\n{col} value counts (after grouping):")
    print(data[col].value_counts())


## using One Hot Encoding for handling categorical data
data = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'], prefix=['protocol_type', 'service', 'flag'])

# Print new dataset
print(f"New Dataset contains {data.shape[0]} rows and {data.shape[1]}columns")

print(data.head())

# Convert boolean values to numeric
data = data.astype(int)

# Covert data to numpy arrays
X = np.array(data)
print(data[:5])

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (Keep 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Check PCA shape
print(f"PCA Output Shape: {X_pca.shape}")
print('Explained variance ratio:', pca.explained_variance_ratio_.sum())
# Print the top components and how much variance they explain
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

# Apply t-SNE for 2D visualization
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print(f"t-SNE Output Shape: {X_tsne.shape}")




# Plot the raw data using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, alpha=0.6, c='steelblue')
plt.title("Raw Data Scatter Plot (t-SNE)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



## Clustering
# k-means Clustering
# run clusterings for different values of k
inertiasAll = []
silhouettesAll = []
for n in range(2, 16):
    print('k-means Clustering for n=', n)
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_pca)
    y_kmeans = kmeans.predict(X_pca)

    # get cluster centers
    kmeans.cluster_centers_

    # evalute
    print('inertia=', kmeans.inertia_)
    silhouette_values = silhouette_samples(X_pca, y_kmeans)
    print('silhouette=', np.mean(silhouette_values))

    inertiasAll.append(kmeans.inertia_)
    silhouettesAll.append(np.mean(silhouette_values))

# Print some statistical data: inertia and silhouette
plt.figure(2)
plt.plot(range(2, 16), silhouettesAll, 'r*-')
plt.ylabel('Silhouette score')
plt.xlabel('Number of clusters k-means')
plt.figure(3)
plt.plot(range(2, 16), inertiasAll, 'g*-')
plt.ylabel('Inertia Score')
plt.xlabel('Number of clusters k-means')
plt.show()

# Find the optimal number of clusters
optimal_clusters_km = range(2, 16)[np.argmax(silhouettesAll)]
print(f"Optimal number of clusters: {optimal_clusters_km}")



# Manually select the optimum number of clusters based on observation and results
selected_clusters_km = 8

# K-Means Clustering with selected number of clusters
kmeans_final = KMeans(n_clusters=selected_clusters_km, random_state=42)
kmeans_final.fit(X_pca)
y_kmeans_final = kmeans_final.predict(X_pca)

# Analyze K-Means cluster sizes
unique_clusters, cluster_counts = np.unique(y_kmeans_final, return_counts=True)

print("\nK-Means Cluster Sizes:")
for label, count in zip(unique_clusters, cluster_counts):
    print(f"Cluster {label}: {count} data points")


# Scatter Plot for K-Means Clustering
# Get the PCA-reduced coordinates of cluster centers
kmeans_cluster_centers = kmeans_final.cluster_centers_

# Scatter Plot for K-Means Clustering

plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans_final, cmap='tab10', s=10, alpha=0.7)
plt.title("K-Means Clusters Visualized via t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()


# --- DBSCAN Clustering ---

eps_values = 2.4
min_samples_values = 30

dbscan = DBSCAN(eps=eps_values, min_samples=min_samples_values)
y_dbscan = dbscan.fit_predict(X_pca)

from collections import Counter

# Count the number of data points in each cluster
cluster_counts = Counter(y_dbscan)

print("\nDBSCAN Cluster Sizes:")
for label, count in sorted(cluster_counts.items()):
    label_name = f"Cluster {label}" if label != -1 else "Noise"
    print(f"{label_name}: {count} data points")


# Cluster stats
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)
silhouette = silhouette_score(X_pca, y_dbscan)

print(f"\n DBSCAN Configuration:")
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
print(f"Silhouette Score: {silhouette:.4f}")

# Visualize cluster sizes for DBSCAN
unique_labels = np.unique(y_dbscan)

dbscan_cluster_labels = [label for label in unique_labels if label != -1]  # exclude noise
dbscan_cluster_sizes = [cluster_counts[label] for label in dbscan_cluster_labels]




# Plot clusters

plt.figure(figsize=(10, 7))
for label in np.unique(y_dbscan):
    label_name = f"Cluster {label}" if label != -1 else "Noise"
    color = 'black' if label == -1 else None
    plt.scatter(X_tsne[y_dbscan == label, 0], X_tsne[y_dbscan == label, 1],
                label=label_name, s=10, alpha=0.6, c=color)

plt.title("DBSCAN Clusters Visualized via t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(markerscale=2, loc='best')
plt.grid(True)
plt.show()


from sklearn.neighbors import NearestNeighbors

# k = min_samples
neighbors = NearestNeighbors(n_neighbors=30)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)

# Sort and plot the distances to the 30th nearest neighbor
sorted_distances = np.sort(distances[:, 29])  # 30th neighbor = index 29
plt.figure(figsize=(8, 4))
plt.plot(sorted_distances)
plt.title("k-Distance Graph (min_samples = 30)")
plt.xlabel("Points sorted by distance")
plt.ylabel("30th Nearest Neighbor Distance")
plt.grid(True)
plt.show()

def evaluate_dbscan_eps_range(X, eps_values, min_samples=30):
    results = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        # Silhouette score only makes sense with 2+ clusters
        if n_clusters > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1

        results.append({
            "eps": eps,
            "clusters": n_clusters,
            "noise": n_noise,
            "silhouette": score
        })

    # Sort by highest silhouette score with preference for more clusters
    best_result = sorted(
        results,
        key=lambda x: (x['silhouette'], x['clusters']),
        reverse=True
    )[0]

    print("\n DBSCAN Parameter Tuning Results:")
    for r in results:
        print(f"eps={r['eps']:.2f} | Clusters={r['clusters']} | Noise={r['noise']} | Silhouette={r['silhouette']:.4f}")

    print(f"\n Best eps: {best_result['eps']} "
          f"(Clusters: {best_result['clusters']}, "
          f"Noise: {best_result['noise']}, "
          f"Silhouette: {best_result['silhouette']:.4f})")

    return best_result

eps_range = np.arange(2.0, 3.6, 0.2)  # try eps values: 2.0, 2.2, ..., 3.4
best_dbscan = evaluate_dbscan_eps_range(X_pca, eps_range, min_samples=30)

# ------------------- FIND ANOMALIES ------------------- #
# Extract anomaly indices and data points
anomaly_indices_dbscan = np.where(y_dbscan == -1)[0]
anomalies_dbscan = data.iloc[anomaly_indices_dbscan]

print(f"Total DBSCAN anomalies detected: {len(anomalies_dbscan)}")

# K-means anomalies

# --- Apply DBSCAN Within Each K-Means Cluster ---

clusterwise_dbscan_anomalies = []

# DBSCAN parameters
eps = 0.8
min_samples = 5

for cluster_label in np.unique(y_kmeans_final):
    # Get indices and points for this cluster
    cluster_indices = np.where(y_kmeans_final == cluster_label)[0]
    cluster_points = X_pca[cluster_indices]

    # Run DBSCAN within this cluster
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    local_labels = dbscan.fit_predict(cluster_points)

    # DBSCAN noise points are labeled -1
    local_anomalies = cluster_indices[local_labels == -1]
    clusterwise_dbscan_anomalies.extend(local_anomalies)

# Convert to numpy array
anomaly_indices_kmeans_dbscan = np.array(clusterwise_dbscan_anomalies)
print(f"Anomalies detected by DBSCAN within K-Means clusters: {len(anomaly_indices_kmeans_dbscan)}")


plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='lightgray', s=10, label='Normal Data')
plt.scatter(X_tsne[anomaly_indices_kmeans_dbscan, 0], X_tsne[anomaly_indices_kmeans_dbscan, 1],
            c='purple', s=25, label='Anomalies (DBSCAN inside K-Means)')
plt.title("Anomalies Detected by DBSCAN within K-Means Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()



# Combine anomaly indices for visualization
anomaly_indices_kmeans = np.array(clusterwise_dbscan_anomalies)


common_anomalies = set(anomaly_indices_dbscan).intersection(set(anomaly_indices_kmeans))
print(f"Common anomalies detected by both DBSCAN and K-Means: {len(common_anomalies)}")



# Highlight Anomaly Percentages
# Total number of data points
total_points = len(X)

# Percentages
pct_dbscan = (len(anomaly_indices_dbscan) / total_points) * 100
pct_kmeans = (len(anomaly_indices_kmeans) / total_points) * 100
pct_common = (len(common_anomalies) / total_points) * 100

# Display results
print(f"Total data points: {total_points}")
print(f"DBSCAN anomalies: {len(anomaly_indices_dbscan)} ({pct_dbscan:.2f}%)")
print(f"K-Means anomalies: {len(anomaly_indices_kmeans)} ({pct_kmeans:.2f}%)")
print(f"Common anomalies: {len(common_anomalies)} ({pct_common:.2f}%)")


# Summarize anomalies and DBSCAN noise within each K-Means cluster
rows = []
unique_clusters = np.unique(y_kmeans_final)

for label in unique_clusters:
    cluster_indices = np.where(y_kmeans_final == label)[0]
    num_total = len(cluster_indices)

    # K-Means anomalies (cluster-wise detection)
    num_kmeans_anomalies = np.sum(np.isin(cluster_indices, anomaly_indices_kmeans))
    pct_kmeans = (num_kmeans_anomalies / num_total) * 100 if num_total > 0 else 0

    # Common anomalies (both KMeans and DBSCAN)
    num_common_anomalies = np.sum(np.isin(cluster_indices, list(common_anomalies)))
    pct_common = (num_common_anomalies / num_total) * 100 if num_total > 0 else 0

    # DBSCAN noise points in the cluster
    dbscan_labels_in_cluster = y_dbscan[cluster_indices]
    num_noise = np.sum(dbscan_labels_in_cluster == -1)
    pct_noise = (num_noise / num_total) * 100 if num_total > 0 else 0

    rows.append({
        "Cluster": label,
        "Total Points": num_total,
        "K-Means Anomalies": f"{num_kmeans_anomalies}/{num_total} ({pct_kmeans:.2f}%)",
        "Common Anomalies": f"{num_common_anomalies}/{num_total} ({pct_common:.2f}%)",
        "DBSCAN Noise in Cluster": f"{num_noise}/{num_total} ({pct_noise:.2f}%)"
    })

# Display as DataFrame
df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))


# 2. K-Means Anomalies Breakdown by DBSCAN Clusters
kmeans_anomaly_dbscan_labels = y_dbscan[anomaly_indices_kmeans]
kmeans_anomaly_dbscan_distribution = Counter(kmeans_anomaly_dbscan_labels)

print("\n K-Means Anomalies distributed over DBSCAN Clusters:")
for label, count in sorted(kmeans_anomaly_dbscan_distribution.items()):
    name = f"Cluster {label}" if label != -1 else "Noise"
    print(f"{name}: {count} K-Means anomalies")

# 3. Common Anomalies Breakdown by DBSCAN Clusters
common_anomalies = np.array(list(set(anomaly_indices_kmeans).intersection(set(anomaly_indices_dbscan))))
common_anomaly_dbscan_labels = y_dbscan[common_anomalies]
common_anomaly_dbscan_distribution = Counter(common_anomaly_dbscan_labels)

print("\n Common Anomalies distributed over DBSCAN Clusters:")
for label, count in sorted(common_anomaly_dbscan_distribution.items()):
    name = f"Cluster {label}" if label != -1 else "Noise"
    print(f"{name}: {count} common anomalies")

# Plotting the anomalies
plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='lightgray', s=10, label='Normal Data', alpha=0.4)
plt.scatter(X_tsne[anomaly_indices_kmeans, 0], X_tsne[anomaly_indices_kmeans, 1],
            c='blue', s=20, label='K-Means Anomalies', alpha=0.5)
plt.scatter(X_tsne[anomaly_indices_dbscan, 0], X_tsne[anomaly_indices_dbscan, 1],
            c='red', s=20, label='DBSCAN Anomalies', alpha=0.7)

# Common anomalies (intersection)
common_indices_tsne = list(common_anomalies)
plt.scatter(X_tsne[common_indices_tsne, 0], X_tsne[common_indices_tsne, 1],
            c='black', s=25, label='Common Anomalies', alpha=1, marker='x')


# Add titles, labels, and legend
plt.title("Detected Anomalies by DBSCAN and K-Means")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()


common_anomalies = np.array(list(set(anomaly_indices_kmeans).intersection(set(anomaly_indices_dbscan))))

def clusterwise_feature_insight(data, common_anomalies, y_kmeans_final, top_n=5):
    df = data.copy()
    df['is_anomaly'] = 0


    df.iloc[common_anomalies, df.columns.get_loc('is_anomaly')] = 1

    cluster_labels = np.unique(y_kmeans_final)
    all_results = {}

    for cluster in cluster_labels:
        cluster_indices = np.where(y_kmeans_final == cluster)[0]
        cluster_df = df.iloc[cluster_indices].copy()

        if cluster_df['is_anomaly'].sum() == 0:
            continue

        normal = cluster_df[cluster_df['is_anomaly'] == 0]
        anomalies = cluster_df[cluster_df['is_anomaly'] == 1]

        normal_mean = normal.mean()
        anomaly_mean = anomalies.mean()

        diff = (anomaly_mean - normal_mean).abs()
        diff = diff.drop(labels='is_anomaly', errors='ignore')  # Exclude helper column from the assessment
        top_diff = diff.sort_values(ascending=False).head(top_n)

        all_results[cluster] = top_diff

        print(f"\nCluster {cluster} - Top {top_n} Differentiating Features")
        print(top_diff)

        # Plot distributions
        for feature in top_diff.index:
            if cluster_df[feature].nunique() <= 1:
                print(f" Skipping {feature} in Cluster {cluster} (0 variance)")
                continue

            plt.figure(figsize=(10, 4))
            sns.kdeplot(data=cluster_df, x=feature, hue="is_anomaly", common_norm=False)
            plt.title(f"Cluster {cluster} - {feature} Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return all_results



# Run the clusterwise feature insight function
clusterwise_feature_insight(data, common_anomalies, y_kmeans_final, top_n=2)


# Which protocol types are more common in anomalies?
# Step 1: Identify protocol type columns (from one-hot encoding)
protocol_cols = [col for col in data.columns if col.startswith("protocol_type_")]

# Step 2: Get only the common anomalies' rows
common_anomalies_df = data.iloc[common_anomalies]

# Step 3: Sum each protocol column and sort
protocol_freq = common_anomalies_df[protocol_cols].sum().sort_values(ascending=False)

# Step 4: Print results
print("\n Protocol frequency in Common Anomalies:")
print(protocol_freq)