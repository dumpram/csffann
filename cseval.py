import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import imgutils as utils
from PIL import Image, ImageFile
from skimage import measure
import pdb

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
    xout = tf.matmul(h, weights[-1])
    
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

    phi = np.random.randn(M, N)
    phi[M - 1, :] = 1.0 / N;
    (Xs, ys) = utils.get_img_dataset(B, M, phi, 'dataset')

    X = tf.placeholder("float", shape=[1, N])
    Y = tf.placeholder("float", shape=[1, M])
    # NN construction
    xhat = forwardprop_multilayer(Y, [M, 256, 256, 256, 128, N])
    #block = tf.reshape(xhat, [B, B])
    cost = tf.reduce_mean(tf.square(tf.subtract(xhat, X)))
    #l1 = tf.norm(dct2(block), ord=1)
    #cost = tf.add(cost, 0.01 * l1)
    updates = tf.train.AdamOptimizer(0.0001).minimize(cost)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, 'models/model.ckpt')

    # Test image
    (Xs, ys) = utils.get_img_measurements(B, M, phi, 'test/cameraman.tif')

    Xsrek = np.zeros(Xs.shape)
    for i in range(len(Xs[:, 1])):
    	Xsrek[i, :] = sess.run(xhat, feed_dict=
    		{Y: np.reshape(ys[i, :], (1, M))})

    image = utils.get_img_from_dct_blocks(szx, szy, B, Xsrek, ys)
    image = (image - np.min(image))
    image = image * 255 / np.max(image)
    orig  = utils.get_img_from_dct_blocks(szx, szy, B, Xs, ys)

    print("PSNR: " + str(utils.psnr(orig, image)))

    plt.imshow(image, cmap='gray')
    plt.figure()
    plt.imshow(orig, cmap='gray')
    plt.show()

    #saver.save(sess, 'models/model.ckpt')
    #sess.close()

if __name__ == "__main__":
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	main()