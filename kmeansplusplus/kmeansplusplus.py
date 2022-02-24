import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')


def euclidian_distance(a, b):
    return np.sqrt(sum((a - b) ** 2))


def init_centroid(centroids, data, distance=euclidian_distance):
    distances = [min([distance(point, x) for point in centroids]) for x in data]
    # normalize distances so that their sum is 1 -> probability law
    probabilities = distances / sum(distances)
    # chose new centroid according to the probability distribution defined above
    new_centroid = np.random.choice(len(probabilities), p=probabilities)
    return data[new_centroid]


def init_centroids(cluster_nb):
    centroid0 = data[np.random.randint(len(data))]
    centroids = np.array([centroid0])
    for i in range(cluster_nb - 1):
        centroids = np.append(centroids, [init_centroid(centroids[-1], data)], axis=0)
    return centroids


def assign_points(data, centroids, distance=euclidian_distance):
    cluster_assignments = [-1 for _ in range(len(data))]
    for i, x in enumerate(data):
        distances = [distance(x, point) for point in centroids]
        cluster_assignments[i] = np.argmin(distances)
    return cluster_assignments


def update_clusters(data, centroids, cluster_assignments):
    for i in range(len(centroids)):
        cluster_idxr = np.where(cluster_assignments == np.full(len(cluster_assignments), i))
        centroids[i, 0] = np.mean(data[cluster_idxr, 0])
        centroids[i, 1] = np.mean(data[cluster_idxr, 1])
    return centroids


def k_means_iteration(data, centroids):
    cluster_assignments = assign_points(data, centroids)
    centroids = update_clusters(data, centroids, cluster_assignments)
    return centroids, cluster_assignments


def k_means(data, nb_clusters, iterations=20):
    centroids = init_centroids(nb_clusters)

    for i in range(iterations):
        centroids, cluster_assignments = k_means_iteration(data, centroids)
        plot = sns.scatterplot(x=data[:, 0], y=data[:, 1], c=cluster_assignments)
        plot.figure.savefig(f"k_means_{nb_clusters}_it_{i}.png")
        plt.scatter(centroids[:, 0], centroids[:, 1], color='r')
        print(centroids)
        #inertia = compute_inertia(data, centroids, cluster_assignments)

if __name__ == "__main__":
    DATA_PATH = "data/data.npy"
    CENTROID_NB = 3

    data = np.load(DATA_PATH)
    k_means(data, CENTROID_NB)

