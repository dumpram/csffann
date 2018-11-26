import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
from PIL import Image, ImageFile
from skimage import measure
import pdb

# User utilities
import imgutils as utils
import common as common

def init_weight_normal(shape):
    weights = tf.random_normal(shape, stddev=0.04)
    return tf.Variable(weights)

def init_bias_normal(shape):
    bias = tf.random_normal(shape, stddev=0.04)
    return tf.Variable(bias)

def init_weight_xavier(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))

def init_bias_xavier(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))

def init_weight_he(shape):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.Variable(initializer(shape))

def init_bias_he(shape):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.Variable(initializer(shape))

def forwardprop_multilayer(xin, sizes):
    weights = [None] * (len(sizes) - 1) 
    biases = [None] * (len(sizes) - 2)
    # Init weights
    for i in range(len(weights)):
        weights[i] = init_weight_normal((sizes[i], sizes[i + 1]))
    # Init biases
    for i in range(len(biases)):
        biases[i] = init_bias_normal((1, sizes[i + 1]))
    
    # Forward propagation
    h = xin
    for i in range(len(weights) - 1):
        h = tf.matmul(h, weights[i])
        h = tf.add(h, biases[i])
        h = tf.nn.relu(h)
    xout = tf.matmul(h, weights[-1], name='xhat')
    
    return xout

def dct2(a):
    n = 8
    dctmat = tf.spectral.dct(tf.eye(n))
    return tf.matmul(tf.matmul(dctmat, a), tf.transpose(dctmat))

def main():
	# Image parameters
    szx = 256
    szy = 256
    B = 8
    N = B * B
    M = int(N / 2)

    # Measurement matrix
    phi = common.get_measurement_matrix_with_embedded_average(M, N, seed=1367)
    # Create dataset
    (xs, ys) = utils.get_img_dataset(B, M, phi, 'dataset')

    X = tf.placeholder("float", shape=[1, N])
    Y = tf.placeholder("float", shape=[1, M], name='Y')
    # NN construction
    xhat = forwardprop_multilayer(Y, [M, 256, 256, 256, 128, N])
    cost = tf.reduce_mean(tf.square(tf.subtract(xhat, X)))
    updates = tf.train.AdamOptimizer(0.0001).minimize(cost)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create log for tensorboard
    writer = tf.summary.FileWriter("logs/sess.log", sess.graph)
    
    # Train NN
    for epoch in range(120):
        train_cost = 0.0
        for i in range(len(xs[:, 1])):
            sess.run(updates, 
                feed_dict={X: np.reshape(xs[i, :], (1, N)), 
                           Y: np.reshape(ys[i, :], (1, M))})
            
            train_cost += sess.run(cost, 
                feed_dict={X: np.reshape(xs[i, :], (1, N)), 
                           Y: np.reshape(ys[i, :], (1, M))})
        
        print("Epoch: %d, train cost: %lf" % (epoch, 
                                                train_cost / len(xs[:, 1])))

    # Load test image
    (xs, ys) = utils.get_img_measurements(B, M, phi, 'test/cameraman.tif')

    xsrek = np.zeros(xs.shape)
    for i in range(len(xs[:, 1])):
    	xsrek[i, :] = sess.run(xhat, feed_dict=
    		{Y: np.reshape(ys[i, :], (1, M))})

    image = utils.get_img_from_dct_blocks(szx, szy, B, xsrek, ys)
    image = (image - np.min(image))
    image = image * 255.0 / np.max(image)
    orig  = utils.get_img_from_dct_blocks(szx, szy, B, xs, ys)

    print("PSNR: " + str(utils.psnr(orig, image)))

    plt.imshow(image, cmap='gray')
    plt.figure()
    plt.imshow(orig, cmap='gray')
    plt.show()

    saver.save(sess, 'models/model.ckpt')
    sess.close()

if __name__ == "__main__":
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	main()