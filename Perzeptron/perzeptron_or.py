import matplotlib.pyplot as plt
import numpy as np


def get_dataset():
    x = np.array([[0,0], [1,0], [0,1], [1,1]])
    y = np.array([0, 1, 1, 1])
    return x, y


class Perceptron():

    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.w  = []
        self.lr = lr
    
    def train(self, x, y):
        n, dim = x.shape
        self.w = np.random.uniform(-1, 1, (dim, 1))
        error = 0.0
        for epoch in range(self.epochs):
            choice = np.random.choice(n)
            x_i = x[choice]
            y_i = y[choice]
            y_hat = self.predict(x_i)
            if y_hat != y_i:
                error += 1
                self.update_weights(x_i, y_i, y_hat)
        print("Train Error: ", error / y.shape[0] )

    def update_weights(self, x, y, y_hat):
        for i in range(self.w.shape[0]):
            delta_w_i = self.lr * (y -y_hat) * x[i]
            self.w[i] += delta_w_i
    
    def test(self, x, y):
        y_pred = np.array([self.predict(x_i) for x_i in x])
        acc = sum(1 for y_p, y_i in zip(y_pred, y) if y_p == y_i) / y.shape[0]
        print("Acc:", acc)
        return acc

    def predict(self, x):
        input_signal = np.dot(self.w.T, x)
        output_signal = self.activation(input_signal)
        return output_signal
    
    def activation(self, signal):
        if signal >  0:
            return 1
        else: 
            return 0

x, y = get_dataset()
lr = 1.0 
epochs = 10
p = Perceptron(epochs=epochs, lr=lr)
p.train(x, y)
p.test(x, y)
