import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import imgutils as utils
from PIL import Image, ImageFile

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
	# Image parameters
    szx = 256
    szy = 256
    B = 8
    N = B * B
    M = int(N / 2)

    phi = np.random.randn(M, N)
    (Xs, ys) = utils.get_img_dataset(B, M, phi, 'dataset')

    X = tf.placeholder("float", shape=[1, N])
    Y = tf.placeholder("float", shape=[1, M])

    # NN construction
    xhat = forwardprop_multilayer(Y, [M, 128, 128, 64, N])

    cost = tf.reduce_mean(tf.square(tf.subtract(xhat, X)))
    updates = tf.train.AdamOptimizer(0.0005).minimize(cost)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create log for tensorboard
    writer = tf.summary.FileWriter("logs/sess.log", sess.graph)
    for epoch in range(20):
        train_cost = 0.0
        for i in range(len(Xs[:, 1])):
            sess.run(updates, 
                feed_dict={X: np.reshape(Xs[i, :], (1, N)), 
                           Y: np.reshape(ys[i, :], (1, M))})
            
            train_cost += sess.run(cost, 
                feed_dict={X: np.reshape(Xs[i, :], (1, N)), 
                           Y: np.reshape(ys[i, :], (1, M))})
        
        print("Epoch: %d, train cost: %lf" % (epoch, 
                                                train_cost / len(Xs[:, 1])))

    # Test image
    (Xs, ys) = utils.get_img_measurements(B, M, phi, 'test/cameraman.tif')

    Xsrek = np.zeros(Xs.shape)
    for i in range(len(Xs[:, 1])):
    	Xsrek[i, :] = sess.run(xhat, feed_dict=
    		{Y: np.reshape(ys[i, :], (1, M))})

    image = utils.get_img_from_dct_blocks(szx, szy, B, Xsrek)

    plt.imshow(image)
    plt.show()

    saver.save(sess, 'models/model.ckpt')
    sess.close()

if __name__ == "__main__":
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	main()