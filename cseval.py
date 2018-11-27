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

def main():
	# Image parameters
    B = 8
    N = B * B
    M = int(N / 2)

    # Measurement matrix
    phi = common.get_measurement_matrix_with_embedded_average(M, N, seed=1367)

    # Restore model
    saver = tf.train.import_meta_graph('models/model.ckpt.meta')
    Y = tf.get_default_graph().get_tensor_by_name('Y:0')
    xhat = tf.get_default_graph().get_tensor_by_name('xhat:0')

    sess = tf.Session()
    saver.restore(sess, 'models/model.ckpt')

    # Load test image
    (Xs, ys, szx, szy) = \
        utils.get_img_measurements(B, M, phi, 'test/cameraman.tif')

    Xsrek = np.zeros(Xs.shape)
    # Reconstruct image blocks
    for i in range(len(Xs[:, 1])):
    	Xsrek[i, :] = sess.run(xhat, feed_dict=
    		{Y: np.reshape(ys[i, :], (1, M))})

    image = utils.get_img_from_blocks(szx, szy, B, Xsrek, ys)
    image = image.clip(min=0.0, max=255.0)
    orig  = utils.get_img_from_blocks(szx, szy, B, Xs, ys)
    print("PSNR: " + str(utils.psnr(orig, image)))

    plt.imshow(image, cmap='gray')
    plt.figure()
    plt.imshow(orig, cmap='gray')
    plt.show()

    sess.close()

if __name__ == "__main__":
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	main()