import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

# plt.scatter(x, y)
# plt.show()

X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, .6], [9, 10]])
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroid, labels)
coolors = ['g.', 'r.','c.']
for i in range(len(x)):
    print("coorrdinates:", X[i], 'Label:', labels[i])
    plt.plot(X[i][0], X[i][1], coolors[labels[i]])

plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.show()
