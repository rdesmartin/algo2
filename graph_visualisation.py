import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from kmeansplusplus.kmeansplusplus import calculate_dist


THRESHOLD = 0.5

if __name__ == "__main__":
    DATA_PATH = "kmeansplusplus/data/data.csv"
    data = pd.read_csv(DATA_PATH).values
    data = data[:500, :]
    adjacency_matrix = {i: set() for i in range(data.shape[0])}

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j:
                continue
            if calculate_dist(data[i], data[j]) >= THRESHOLD:
                adjacency_matrix[i].add(j)
                adjacency_matrix[j].add(i)

    plot = sns.scatterplot(x=data[:, 0], y=data[:, 1])
    for i in adjacency_matrix:
        x1 = data[i, 0]
        y1 = data[i, 1]
        for j in adjacency_matrix[i]:
            x2 = data[j, 0]
            y2 = data[j, 1]
            plt.plot([x1, x2], [y1, y2], 'k-')
    plot.figure.savefig("graph.png")


