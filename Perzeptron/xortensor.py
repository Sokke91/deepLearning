import tensorflow as tf
import numpy as np

def getDataset():
    x = np.array([[0,0],[1,0],[0,1],[1,1]])
    y = np.array([[0], [1], [1], [0]]).astype(np.float32)
    return x, y

x, y = getDataset()

x_train, y_train = x, y 
x_test, y_test = x, y 

features = 2 
classes = 2

target = 1 

nodes = [features, 2, target]  # input, hidden, output

train_size  = x_train.shape[0]
test_size = x_test.shape[0]

epochs = 10

x = tf.placeholder(dtype=tf.float32, shape=[], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[], name="y")
