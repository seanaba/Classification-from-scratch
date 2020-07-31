from random import random

class LinearRegression:

    def __init__(self, learning_rate=0.01, num_iteration=10000, threshold=0.0001):
        """LinearRegression constructor
        """
        self.x = []
        self.y = []
        self.weight = random()
        self.bias = random()
        self.learning_rate = learning_rate
        self.number_iteration = num_iteration
        self.threshold = threshold

    def fit(self, x, y):
        """Training the model
        :param x: feature
        :param y: label
        :return: None
        """
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.weight, self.bias = \
            self.train(self.x, self.y, self.weight, self.bias,
                       self.learning_rate, self.threshold, self.number_iteration)

    @staticmethod
    def cost_function(x, y, w, b):
        """
        claculates the cost function
        :param x: feature
        :param y: label
        :param w: weight
        :param b: bias
        :return: cost of new parameters
        """
        total_cost = 0.0
        for i in range(len(x)):
            total_cost += (y[i] - (w*x[i] + b))**2
        return total_cost / (2*len(x))

    @staticmethod
    def update_weights(x, y, w, b, lr):
        """
        updates weight and bias
        :param x: feature
        :param y: label
        :param w: weight
        :param b: bias
        :param lr: learning rate
        :return: updated weight and bias
        """
        w_tot = b_tot = 0
        for i in range(len(x)):
            w_tot += (((w * x[i]) + b) - y[i]) * x[i]
            b_tot += (((w * x[i]) + b) - y[i])
        w = w - (lr * (1 / len(x)) * w_tot)
        b = b - (lr * (1 / len(x)) * b_tot)
        return w, b

    def train(self, x, y, w, b, lr, thre, num_iters):
        """
        train weights
        :param x: feature
        :param y: label
        :param w: weight
        :param b: bias
        :param lr: learning rate
        :param thre: threshold of stopping criteria
        :param num_iters: maximum number of iteration
        :return: weights
        """
        cost_lst = []
        for i in range(int(num_iters)):
            weight, bias = self.update_weights(x, y, w, b, lr)
            cost = self.cost_function(x, y, weight, bias)
            cost_lst.append(cost)
            if cost_lst and abs(cost_lst[-1] - cost) < thre:
                break
        return weight, bias

    def predict(self, x_te):
        """
        predicts labels
        :param x_te: features in list format
        :return: labels
        """
        res = []
        for x_val in x_te:
            res.append(round((self.weight * x_val) + self.bias, 1))
        return res


if __name__ == '__main__':
    x_train = [1, 3.1, 4.3, 7.04, 2.03, 5.2]
    y_train = [1, 2.9, 3.9, 7.1, 1.98, 5.31]
    x_test = [-1.02, 1.89, 12.4]
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))