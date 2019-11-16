from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf
import numpy as np
import os

# Dataset Vars
dir_path = "C:/Users/jonas/Documents/Development/playground/DeepLearning/Data"
mnist = input_data.read_data_sets(dir_path, one_hot=True)
train_size = mnist.train.num_examples
test = mnist.test.num_examples
features = 784
classes = 10
# Training HyperParameter
epochs = 50
train_batch_size = 128 # n^2
learning_rate = 1e-3 
#Model HyperParameter
layer_nodes = [features, 250, classes]
stddev = 0.100
bias_weight_init= 0.0 
#Helper Vars
test_batch_size = 1000
train_mini_batches = int(train_size/train_batch_size) + 1 
test_mini_batches = int(test_size/test_batch_size) + 1
train_errors, test_errors = [], []
train_accs, test_accs = [], []

#Tf Placeholders for input and output of the dnn
x = tf.placeholder(dtype=tf.float32, shape=[none, features], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[none, classes], name="y")

# Weights Vars
W1 = tf.Variable(tf.truncated_normal([layer_nodes[0], layer_nodes[1]], stddev=stddev), name="W1") # Input to Hidden
W2 = tf.Variable(tf.truncated_normal([layer_nodes[1], layer_nodes[2]], stddev=stddev), name="W1") # Hidden to Output

# Biases
b1 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[1]]), name="b1") # Hidden
b2 = tf.Variable(tf.constant(bias_weight_init, shape=[layer_nodes[2]]), name="b2") # Output

# The DNN Model inputlayer -> hiddenlayer -> outputlayer
def nn_model(x):
    input_layer_dict = {"weights":W1, "biases": b1}
    hidden_layer_dict = {"weights": W2, "biases": b2}
    input_layer = x
    hidden_layer_in = tf.add(tf.matmul(input_layer, input_layer_dict["weights"]), input_layer_dict["biases"])
    hiden_layer_out = tf.nn.relu(hidden_layer_in)
    output_layer = tf.add(tf.matmul(hiden_layer_out, hidden_layer_dict["weights"]), hidden_layer_dict["biases"])
    return output_layer

def nn_run():
    # Tensorflow Ops 
    pred = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_result = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuarancy = tf.reduce_mean(tf.cast(correct_result, tf.float32))

    # Start Tensorflow Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Training
        for epoch in range(epochs):
            train_acc, train_loss = 0.0, 0.0
            test_acc, test_loss = 0.0, 0.0
            #Train Weights
            