import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, num_iteration=10000, threshold=0.0001):
        """LinearRegression constructor
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
        :param y: output
        :return: None
        """
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.weight = \
            self.train(self.x, self.y, self.weight,
                       self.learning_rate, self.threshold, self.number_iteration)

    @staticmethod
    def cost_function(x, y, w):
        """
        claculates the cost function
        :param x: feature set
        :param y: output
        :param w: weight
        :return: cost of new parameters
        """
        return np.sum((x.dot(w) - y) ** 2) / (2*len(y))

    @staticmethod
    def update_weights(x, y, w, lr):
        """
        updates weight and bias
        :param x: feature set
        :param y: output
        :param w: weight
        :param lr: learning rate
        :return: updated weight and bias
        """
        loss = x.dot(w) - y
        gradient = x.T.dot(loss) / len(y)
        return w - (lr * gradient)

    def train(self, x, y, w, lr, thre, num_iters):
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
        predicts regression output
        :param x_te: features in list format
        :return: regression output
        """
        x_te = np.array(x_te)
        xt_b = np.ones(shape=x_te.shape[0]).reshape(-1, 1)
        x1 = np.concatenate((xt_b, x_te), 1)
        return np.round(x1.dot(self.weight), 2)


if __name__ == '__main__':
    x_train = [[1, 2], [3.1, 4.01], [4.3, 5.9], [7.04, 8.01], [2.03, 2.89], [5.2, 6.9]]
    y_train = [1, 2.9, 3.9, 7.1, 1.98, 5.31]
    x_test = [[-1.02, 0.01], [1.89, 2.75], [12.4, 13.5]]
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))