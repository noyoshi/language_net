#!/usr/bin/env python3.5 

import tensorflow as tf
import numpy as np

from parse_data import create_feature_sets_and_labels

data_sets = ['lists/japanese_list.txt', 'lists/english_list.txt']
train_x, train_y, test_x, test_y = create_feature_sets_and_labels(data_sets)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2

batch_size = 100
# we go through batches of 100 features at a time, so that we do not have to
# load everything into memory at the same time?

# Matrix is height x width
x = tf.placeholder('float', [None,1080])
# imagine a 1 x 784 sized matrix, pixels are all moved into one row from square
# image
y = tf.placeholder('float')
# y is the label of x?

def neural_network_model(data):

    # using a bias because when an input is zero, then neurons do not fire,
    # which is not ideal. this ensures that some neurons will fire even when
    # input is all zeros.

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 - tf.nn.relu(l1)
    # relu is our treshold function (rectified linear is the real name)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 - tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 - tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # learning rate = 0.001, default
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    hm_epochs = 1
    # how many epochs?

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # initializes all of the variables, the session is now live!

        writer = tf.summary.FileWriter("logs/", sess.graph)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # print(batch_x)
                # print(batch_y)
                # print(type(batch_x))
                # print(type(batch_y))
                # print(len(batch_x))
                # print(len(batch_y))


                # chunks through dataset for you... but in the real world we would
                # have to do this on our own. mnist does this automatically
                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        # checks to see whether we are correct or not

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # casts correct to type float

        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)
