import numpy as np
import pandas as pd

import seaborn as sns

from kmeansplusplus import k_means, calculate_dist

def inertia(data, centroids, assignments, distance):
    distances = []
    for i, x in enumerate(data):
        dist = distance(centroids[assignments[i]], x)
        distances.append(dist**2)
    return np.sum(distances)

if __name__ == '__main__':
    NB_CLUSTERS = 13
    DATA_PATH = "data/data.csv"
    data = pd.read_csv(DATA_PATH)
    inertia_list = []

    for i in range(2, NB_CLUSTERS + 1):
        centroids, assignments = k_means(data.values, i)
        inertia_value = inertia(data.values, centroids, assignments, calculate_dist)
        inertia_list.append(inertia_value)
        print(f"Inertia for {i} clusters: {inertia_value}")

    print(inertia_list)
    plot = sns.lineplot(x=range(len(inertia_list)), y=inertia_list)
    plot.figure.savefig("k_means_elbow.png")
