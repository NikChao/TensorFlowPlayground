import csv
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Random seed should remain constant so that train_test_split is consistent
# during development. Once you move to prod you can make it properly random
RANDOM_SEED = 42

def classification_file_to_matrix(filename):
    """ Reads file with indexed classifications to computable matrices """
    rows = []
    with open(filename, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=',', quotechar='|'):
            rows.append([x.replace(' ', '') for x in row])

    rows = rows[1:]
    all_X = [ x[0: len(x) - 1] for x in rows]
    all_Y = [ x[len(x) - 1] for x in rows]

    # csv reader reads everything as a string
    # turn it all into numbers
    all_Y = [int(y) for y in all_Y]
    all_X = np.array([[float(x_1) for x_1 in x] for x in all_X])
    num_labels = len(np.unique(all_Y))
    all_Y = np.eye(num_labels)[all_Y]

    return train_test_split(all_X, all_Y, test_size=0.33,
                            random_state=RANDOM_SEED)



def continuous_file_to_matrix(filename):
    """ Reads file with continuous results column to computable matrices """
    rows = []
    with open(filename, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=',', quotechar='|'):
            rows.append([x.replace(' ', '') for x in row])

    rows = rows[1:]
    all_X = [ x[0: len(x) - 1] for x in rows]
    all_Y = [ x[len(x) - 1] for x in rows]

    # turn csv read strings into swaggy numbers
    all_Y = [float(y) for y in all_Y]
    all_X = np.array([[float(x_1) for x_1 in x] for x in all_X])

    return train_test_split(all_X, all_Y, test_size=0.33,
                            random_state=RANDOM_SEED)



def init_weights(shape):
    """ Weights Initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)



def forwardprop(X, w_1, w_2):
    """
    Forward-propagation
    Important: yhat is not softmax since tf.softmax_cross_entropy_with_logits()
    does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # the \varphi function
    return yhat



def classification_optimize(filename):
    train_X, test_X, train_y, test_y = classification_file_to_matrix(filename)

    x_size = train_X.shape[1]  # Number of input nodes
    h_size = 256               # Number of hidden nodes (can change)
    y_size = train_y.shape[1]  # Number of outcomes

    # Symbols
    X = tf.placeholder('float', shape=[None, x_size])
    y = tf.placeholder('float', shape=[None, y_size])

    # Weight initialization
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward prop
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGF
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1],
                                         y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                sess.run(predict, feed_dict={X: train_X,
                                                             y: train_y}))

        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_X,
                                                             y: test_y}))

        print(f'Epoch = {epoch}, train accuracy = {100. * train_accuracy, }' +
              f'test accuracy = {100. * test_accuracy}')


    sess.close()



def continuous_optimize(filename):
    train_X, test_X, train_y, test_y = continuous_file_to_matrix(filename)

    x_size = train_X.shape[1]  # Number of input nodes
    h_size = 256               # Number of hidden nodes
    y_size = 1                 # Number of outcomes (just one continuous)

    X = tf.placeholder('float', shape=[None, x_size])
    y = tf.placeholder('float', shape=[None, y_size])

    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    print(predict)


def main():
    filename = input('filename?')
    continuous_optimize(filename)


if __name__ == '__main__':
    main()
