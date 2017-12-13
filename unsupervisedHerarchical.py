import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1, 1], [5, 5],[3,10]]
x, y = make_blobs(n_samples=200, centers=centers, cluster_std=0.3)
plt.scatter(x[:, 0], x[:, 1])
# plt.show()
ms = MeanShift()
ms.fit(x)
labels = ms.labels_
centers = ms.cluster_centers_
nClusters = len(np.unique(labels))

print(centers)
print(nClusters)
colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=10)

plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.show()
