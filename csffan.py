import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_sparse_dataset(start, size, A, N, M, K):
    ys = np.empty([M, size - start])
    xs = np.empty([N, size - start])
    idx = 0
    for i in range(start, start + size - 1):
        (x, y) = get_sparse_signal(i, A, N, K)
        ys[:, idx] = y;
        xs[:, idx] = x;
        idx += 1
    return (xs, ys)
    
def get_sparse_signal(seed, A, N, K):
    np.random.seed(seed)
    x = np.concatenate((np.zeros([N - K, 1]), np.random.rand(K, 1)), axis=None)
    x = np.random.permutation(x)
    y = np.matmul(A, x)
    return (x, y)

def init_weight(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(xin, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(xin, w_1))  # The \sigma function
    xout = tf.matmul(h, w_2)  # The \varphi function
    return xout

def init_bias(shape):
    """ Bias initialization """
    bias = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(bias)

def forwardprop_multilayer(xin, sizes):
    weights = [None] * (len(sizes) - 1) 
    biases = [None] * (len(sizes) - 1)
    for i in range(len(weights)):
        weights[i] = init_weight((sizes[i], sizes[i + 1]))
        biases[i] = init_bias((1, sizes[i + 1]))
    h = xin
    for i in range(len(weights) - 1):
        h = tf.matmul(h, weights[i])
        h = tf.add(h, biases[i])
        h = tf.nn.sigmoid(h)

    xout = tf.matmul(h, weights[-1])
    return xout

M = 8
N = 20
K = 2

A = np.random.rand(M, N)

(xs, ys) = get_sparse_dataset(1, 1000, A, N, M, K)

X = tf.placeholder("float", shape=[1, N])
Y = tf.placeholder("float", shape=[1, M])

xhat = forwardprop_multilayer(Y, [M, 100, 100, 40, N])

cost = tf.nn.l2_loss(tf.subtract(xhat, X))
cost = tf.add(cost, tf.norm(xhat, ord=1) * 0.01)
updates = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter("logs/sess.log", sess.graph)

for epoch in range(1500):
    train_cost = 0.0
    for i in range(len(xs[1, :])):
        sess.run(updates, feed_dict={X: np.reshape(xs[:, i], (1, N)), Y: np.reshape(ys[:, i], (1, M))})
        train_cost += np.mean(sess.run(cost, feed_dict={X: np.reshape(xs[:, i], (1, N)), Y: np.reshape(ys[:, i], (1, M))}))
    
    print('Epoch ' + str(epoch) + ' train cost: ' + str(train_cost / len(xs)))

#(xtest, ytest) = get_sparse_dataset(1000, 2000, A, N, M, K)

(tx, ty) = get_sparse_signal(202020, A, N, K)
err = np.mean(sess.run(cost, feed_dict={X: np.reshape(tx, (1, N)), Y: np.reshape(ty, (1, M))}))
print(err)

xrek = sess.run(xhat, feed_dict={Y: np.reshape(ty, (1, M))})


plt.stem(xrek[0], linefmt='r--', markerfmt='o')
plt.stem(tx)
plt.show()

import pdb
pdb.set_trace()

sess.close()


    
