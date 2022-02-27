from typing import Iterable
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def euclidian_distance(a, b):
    return np.sqrt(sum((a - b) ** 2))

def calculate_dist(d1, d2):
    age_dist = abs(float(d1[0]) - float(d2[0])) * 0.8
    hypertension_dist = abs(float(d1[1]) - float(d2[1])) * 0.8
    heart_disease_dist = abs(float(d1[2]) - float(d2[2])) * 0.8
    smoking_status_dist = abs(float(d1[6]) - float(d2[6])) * 0.8
    avg_glucose_level_dist = abs(float(d1[4]) - float(d2[4])) * 0.5
    bmi_dist = abs(float(d1[5]) - float(d2[5])) * 0.5
    residence_type_dist = abs(float(d1[3]) - float(d2[3])) * 0.2
    distance = age_dist + hypertension_dist + heart_disease_dist + residence_type_dist + avg_glucose_level_dist + bmi_dist + smoking_status_dist
    return distance
    
def init_centroid(centroids, data, distance=calculate_dist):
    distances = np.array([min([distance(point, x) for point in centroids]) for x in data])
    # normalize distances so that their sum is 1 -> probability law
    probabilities = distances / sum(distances)
    # chose new centroid according to the probability distribution defined above
    new_centroid = np.random.choice(len(probabilities), p=probabilities)
    return data[new_centroid]


def init_centroids(cluster_nb, data):
    centroid0 = data[np.random.randint(len(data))]
    centroids = np.array([centroid0])
    for i in range(cluster_nb - 1):
        centroids = np.append(centroids, [init_centroid(centroids, data)], axis=0)
    return centroids


def assign_points(data, centroids, distance=calculate_dist):
    cluster_assignments = [-1 for _ in range(len(data))]
    for i, x in enumerate(data):
        distances = [distance(x, point) for point in centroids]
        cluster_assignments[i] = np.argmin(distances)
    return cluster_assignments


def update_clusters(data, centroids, cluster_assignments):
    for i in range(centroids.shape[0]):
        cluster_idxr = np.where(cluster_assignments == np.full(len(cluster_assignments), i))
        for j in range(centroids.shape[1]):
            centroids[i, j] = np.mean(data[cluster_idxr, j])
    return centroids


def k_means_iteration(data, centroids):
    cluster_assignments = assign_points(data, centroids)
    centroids = update_clusters(data, centroids, cluster_assignments)
    return centroids, cluster_assignments


def k_means(data, nb_clusters, iterations=20):
    centroids = init_centroids(nb_clusters, data)
    cluster_assignments = None

    for i in range(iterations):
        centroids, cluster_assignments = k_means_iteration(data, centroids)
        # plot = sns.scatterplot(x=data[:, 0], y=data[:, 1], c=cluster_assignments)
        # plot.figure.savefig(f"k_means_{nb_clusters}_it_{i}.png")
        # plt.scatter(centroids[:, 0], centroids[:, 1], color='r')

    return centroids, cluster_assignments

if __name__ == "__main__":
    DATA_PATH = "data/data.csv"
    CENTROID_NB = 3
    data = pd.read_csv(DATA_PATH)
    k_means(data.values, CENTROID_NB)

