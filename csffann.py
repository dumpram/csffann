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
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    bias = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(bias)

def forwardprop_multilayer(xin, sizes):
    weights = [None] * (len(sizes) - 1) 
    biases = [None] * (len(sizes) - 2)
    # Init weights
    for i in range(len(weights)):
        weights[i] = init_weight((sizes[i], sizes[i + 1]))
    # Init biases
    for i in range(len(biases)):
        biases[i] = init_bias((1, sizes[i + 1]))
    
    # Forward propagation
    h = xin
    for i in range(len(weights) - 1):
        h = tf.matmul(h, weights[i])
        #h = tf.add(h, biases[i])
        h = tf.nn.relu(h)
    xout = tf.matmul(h, weights[-1])
    
    return xout

def main():
    M = 8
    N = 20
    K = 2

    A = np.random.rand(M, N)
    (xs, ys) = get_sparse_dataset(1, 10000, A, N, M, K)

    X = tf.placeholder("float", shape=[1, N])
    Y = tf.placeholder("float", shape=[1, M])

    xhat = forwardprop_multilayer(Y, [M, 100, 100, 60, N])

    cost = tf.reduce_mean(tf.square(tf.subtract(xhat, X)))
    #cost = tf.add(cost, tf.norm(xhat, ord=1) * 0.0005)
    updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.summary.FileWriter("logs/sess.log", sess.graph)

    for epoch in range(250):
        train_cost = 0.0
        for i in range(len(xs[1, :])):
            sess.run(updates, 
                feed_dict={X: np.reshape(xs[:, i], (1, N)), 
                           Y: np.reshape(ys[:, i], (1, M))})
            
            train_cost += sess.run(cost, 
                feed_dict={X: np.reshape(xs[:, i], (1, N)), 
                           Y: np.reshape(ys[:, i], (1, M))})
        
        print("Epoch: %d, train cost: %lf" % (epoch, 
                                                train_cost / len(xs[1, :])))

    (tx, ty) = get_sparse_signal(202020, A, N, K)

    err = sess.run(cost, 
                feed_dict={X: np.reshape(tx, (1, N)), 
                           Y: np.reshape(ty, (1, M))})

    print('Test error: ' + str(err))
    xrek = sess.run(xhat, feed_dict={Y: np.reshape(ty, (1, M))})

    plt.stem(tx)
    plt.stem(xrek[0], linefmt='r--', markerfmt='o')
    plt.show()

    #import pdb
    #pdb.set_trace()

    sess.close()

if __name__ == "__main__":
    main()

# tf.nn.dropout
# he inicijalizacija
# xavier incijalizacija
# relu aktivacija
# truncated normal distribution