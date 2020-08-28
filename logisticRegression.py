import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iteration=10000, threshold=0.0001):
        """LogisticRegression constructor
        """
        self.x = []
        self.y = []
        self.weight = []
        self.learning_rate = learning_rate
        self.number_iteration = num_iteration
        self.threshold = threshold

    def fit(self, x, y):
        """Training the model
        :param x: feature set
        :param y: labels
        :return: None
        """
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.weight = \
            self.train(self.x, self.y,
                       self.learning_rate, self.threshold, self.number_iteration)

    @staticmethod
    def sigmoid(x, w):
        return 1 / (1 + np.exp(-np.dot(x, w)))

    def cost_function(self, x, y, w):
        """
        claculates the cost function
        :param x: feature set
        :param y: label
        :param w: weight
        :return: cost of new parameters
        """
        return (-y * np.log(self.sigmoid(x, w)) - (1 - y) * np.log(1 - self.sigmoid(x, w))).mean()

    def update_weights(self, x, y, w, lr):
        """
        updates weight and bias
        :param x: feature set
        :param y: output
        :param w: weight
        :param lr: learning rate
        :return: updated weight and bias
        """
        error = self.sigmoid(x, w) - y
        gradient = np.dot(x.T, error) / y.shape[0]
        return w - (lr * gradient)

    def train(self, x, y, lr, thre, num_iters):
        """
        train weights
        :param x: feature set
        :param y: output
        :param w: weight
        :param lr: learning rate
        :param thre: threshold of stopping criteria
        :param num_iters: maximum number of iteration
        :return: weights
        """
        x, y = np.array(x), np.array(y).reshape((-1, 1))
        if len(x) == 1:
            x = x.reshape(-1, 1)
        x_b = np.ones(shape=x.shape[0]).reshape(-1, 1)
        x = np.concatenate((x_b, x), 1)
        w = np.random.rand(x.shape[-1], 1)
        cost_lst = []
        for i in range(int(num_iters)):
            weight = self.update_weights(x, y, w, lr)
            cost = self.cost_function(x, y, weight)
            cost_lst.append(cost)
            if cost_lst and abs(cost_lst[-1] - cost) < thre:
                break
        return weight

    def predict(self, x_te):
        """
        predicts logistic regression output
        :param x_te: features in list format
        :return: label
        """
        x_te = np.array(x_te)
        xt_b = np.ones(shape=x_te.shape[0]).reshape(-1, 1)
        x1 = np.concatenate((xt_b, x_te), 1)
        print(self.sigmoid(x1, self.weight))
        return np.round(self.sigmoid(x1, self.weight))


if __name__ == '__main__':
    x_train = [[1, 2], [3.1, 4.01], [4.3, 5.9], [7.04, 8.01], [2.03, 2.89], [5.2, 6.9]]
    y_train = [1, 0, 0, 0, 1, 0]
    x_test = [[-1.02, 0.01], [1.89, 2.75], [12.4, 13.5]]
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))