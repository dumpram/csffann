import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy import fftpack

def get_sparse_dataset(start, size, A, N, M, K):
    ys = np.empty([M, size - start + 1])
    xs = np.empty([N, size - start + 1])
    idx = 0 
    for i in range(start, start + size):
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
        h = tf.add(h, biases[i])
        h = tf.nn.relu(h)
    xout = tf.matmul(h, weights[-1])
    
    return xout

def main():
    # CS parameters
    M = 8 # number of measurements
    N = 20 # number of samples
    K = 2 # number of samples != 0

    # Dataset size
    n = 1000

    # Measurement matrix
    phi = np.random.randn(M, N)
    # Transformation matrix
    psi = fftpack.idct(np.eye(N, N))
    # CS matrix 
    A = np.matmul(phi, psi)

    (Xs, ys) = get_sparse_dataset(1, n, A, N, M, K)

    psi_inv = fftpack.dct(np.eye(N, N))        
    xs = np.zeros(Xs.shape)

    for i in range(n):
        xs[:, i] = np.matmul(psi, Xs[:, i])

    X = tf.placeholder("float", shape=[1, N])
    x = tf.placeholder("float", shape=[1, N])
    y = tf.placeholder("float", shape=[1, M])

    # NN construction
    xhat = forwardprop_multilayer(y, [M, 100, 100, 80, N])
    # Transformation domain
    Xhat = tf.matmul(tf.cast(psi_inv, tf.float32), tf.transpose(xhat)) / (2. * N)
    # Error in time domain
    mse = tf.reduce_mean(tf.square(tf.subtract(x, xhat)))
    # Sparsity in transformation domain
    l1 = tf.norm(Xhat, ord=1) * 10e-3
    # Error in transofrmation domain
    costX = tf.reduce_mean(tf.square(tf.subtract(Xhat, X)))
    # Total cost
    cost = tf.add(mse, l1)

    updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create log for tensorboard
    writer = tf.summary.FileWriter("logs/sess.log", sess.graph)

    for epoch in range(20):
        train_cost = 0.0
        cost_dct = 0.0
        for i in range(n):
            sess.run(updates, 
                feed_dict={X: np.reshape(Xs[:, i], (1, N)),
                           x: np.reshape(xs[:, i], (1, N)), 
                           y: np.reshape(ys[:, i], (1, M))})
            
            train_cost += sess.run(cost, 
                feed_dict={X: np.reshape(Xs[:, i], (1, N)),
                           x: np.reshape(xs[:, i], (1, N)), 
                           y: np.reshape(ys[:, i], (1, M))})

            cost_dct += sess.run(costX, 
                feed_dict={X: np.reshape(Xs[:, i], (1, N)),
                           x: np.reshape(xs[:, i], (1, N)), 
                           y: np.reshape(ys[:, i], (1, M))})
        
        print("Epoch: %d, train cost (time): %lf, cost (dct): %lf" % 
            (epoch, train_cost / n, cost_dct / n))

    # Test sparse signal
    (tX, ty) = get_sparse_signal(202020, A, N, K)
    tx = np.matmul(psi, tX)

    err = sess.run(cost, 
                feed_dict={X: np.reshape(Xs[:, i], (1, N)),
                           x: np.reshape(xs[:, i], (1, N)), 
                           y: np.reshape(ys[:, i], (1, M))})

    print('Test error: ' + str(err))

    # Signal reconstruction
    start = time.time()
    xrek = sess.run(xhat, feed_dict={y: np.reshape(ty, (1, M))})
    duration = time.time() - start;

    print("Reconstruction time: " + str(duration * 1000) + " ms")

    # Display test signal and reconstruction
    plt.stem(tx)
    plt.stem(xrek, linefmt='r--', markerfmt='o')
    plt.show()

    import pdb
    pdb.set_trace()

    saver.save(sess, 'models/model.ckpt')
    sess.close()

if __name__ == "__main__":
    main()