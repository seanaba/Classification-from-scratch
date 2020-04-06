class KNearestNeighbors:

    def __init__(self, k=3):
        """KNN constructor
        :param k: number of neighbors
        """
        self.k = int(k)
        self._training_data = []

    def fit(self, x, y):
        """Training the model
        :param x: features
        :param y: labels
        :return: None
        """
        assert len(x) == len(y)
        self._training_data = [(feats, label) for feats, label in zip(x, y)]

    @staticmethod
    def distance(x1, x2):
        """Euclidean distance between two points
        :param x1: first point
        :param x2: second
        :return: euclidean distance
        """
        return sum([(i-j)**2 for i, j in zip(x1, x2)])**0.5

    def predict(self, x_te):
        """Predict labels for test data
        :param x_te: test data
        :return: x predicted labels
        """
        results = []
        for points in x_te:
            distances = []
            for feat, label in self._training_data:
                distances.append((self.distance(feat, points), label))
            # selecting sorted k points in ascending manner
            distances = sorted(distances, key=lambda x: x[0])[:self.k]
            d = {}
            for item in distances: d[item[1][0]] = d.get(item[1][0], 0) + 1
            results.append([sorted(d.items(), key=lambda x:x[1], reverse=True)[0][0]])
        return results


if __name__ == '__main__':
    x_train = [[2, 2], [-1, 1], [-2, 1], [4.5, 3.5], [3.5, 4.5], [2.5, 3.5]]
    y_train = [[1], [0], [0], [1], [1], [1]]
    x_test = [[4, 5], [-2, 3]]
    clf = KNearestNeighbors()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))
