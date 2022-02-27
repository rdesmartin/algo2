import numpy as np
import pandas as pd
import seaborn as sns

from kmeansplusplus import calculate_dist, k_means


# Compute s(i) — silhouette coefficient or i’th point using below mentioned formula.

def mean_distance_to_cluster(x, cluster_data):
    mean_distance = 0
    for y in cluster_data:
        mean_distance += calculate_dist(x, y)
    mean_distance /= len(cluster_data)
    return mean_distance


def silhouette(data, centroids, assignments):
    score = 0
    for i, x in enumerate(data):
        # compute s(x), the individual silhouette coefficient for a point
        cluster = assignments[i]
        cluster_center = centroids[cluster]
        cluster_idxr = np.where(assignments == cluster)
        cluster_data = data[cluster_idxr]

        # Compute a(x): The average distance of that point with all other points in the
        # same clusters.
        a = mean_distance_to_cluster(x, cluster_data)

        # Compute b(x): The average distance of that point with all the points in the
        # closest cluster to its cluster.
        cluster_distances = [calculate_dist(cluster_center, c2) for c2 in centroids]
        cluster_distances[cluster] = np.inf
        closest_cluster = np.argmin(cluster_distances)
        cluster_idxr = np.where(assignments == closest_cluster)
        cluster_data = data[cluster_idxr]
        b = mean_distance_to_cluster(x, cluster_data)

        # Compute s(x) — silhouette coefficient for i’th point using below-mentioned
        # formula.
        s = (b - a) / np.max((a, b))

        score += s
    score /= data.shape[0]
    return score


if __name__ == '__main__':
    NB_CLUSTERS = 13
    DATA_PATH = "data/data.csv"
    data = pd.read_csv(DATA_PATH)
    silhouette_list = []

    for i in range(2, NB_CLUSTERS + 1):
        centroids, assignments = k_means(data.values, i)
        silhouette_score = silhouette(data.values, centroids, assignments)
        silhouette_list.append(silhouette_score)
        print(f"Silhouette score for {i} clusters: {silhouette_score}")

    print(silhouette_list)
    plot = sns.lineplot(x=range(len(silhouette_list)), y=silhouette_list)
    plot.figure.savefig("k_means_silhouette.png")




