import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.01):
        """KMeans constructor
        :param k: number of clusters
        :param max_iter: maximum iteration
        :param tol: optimization parameter
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers = {}
        self.clusters = {}

    def fit(self, data):
        """
        Training the model
        :param data: dataset
        """
        for i in range(self.k):
            self.cluster_centers[i] = data[np.random.randint(0, len(data))]

        for _ in range(self.max_iter):
            for j in range(self.k):
                self.clusters[j] = []

            for feat in data:
              dist_clusters = [np.linalg.norm(feat - self.cluster_centers[j]) for j in self.cluster_centers]
              cluster_idx = dist_clusters.index(min(dist_clusters))
              self.clusters[cluster_idx].append(feat)

            prev_cluster_centers = self.cluster_centers
            for j in self.cluster_centers:
                self.cluster_centers[j] = np.average(self.clusters[j], axis=0)

            optimized = False
            for j in self.cluster_centers:
                prev_centers, cur_centers = prev_cluster_centers[j], self.cluster_centers[j]
                if np.sum(((cur_centers-prev_centers)/prev_centers)*100) <= self.tol:
                    optimized = True
            if optimized:
                break

    def predict(self, data):
        """
        make prediction based on data
        :param data: test datasets
        :return: the assigned clusters
        """
        results = []
        for feat in data:
            dist = [np.linalg.norm(feat-self.cluster_centers[j]) for j in self.cluster_centers]
            results.append(dist.index(min(dist)))
        return results


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(2)
    kmeans.fit(X)
    print(kmeans.predict([[0, 0], [12, 3]]))