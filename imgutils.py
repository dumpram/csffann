import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image, ImageFile
from scipy import fftpack

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_block(image, x, y, size=8):
	return image[x * size : (x + 1) * size, y * size : (y + 1) * size]

def dct2(a):
    return fftpack.dct(fftpack.dct(
    	a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return fftpack.idct(fftpack.idct(
    	a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def to_vec(a):
	(m, n) = a.shape
	x = np.zeros((1, m * n))
	idx = 0
	for i in range(m):
		for j in range(n):
			x[0, idx] = a[i, j]
			idx += 1
	return x

def to_mat(a, size):
	x = np.zeros((size, size))
	idx = 0
	for i in range(size):
		for j in range(size):
			x[i, j] = a[idx]
			idx += 1
	return x


def get_img_dataset(B, M, phi, path):
	files = os.listdir(path)
	Xs = np.array([])
	ys = np.array([])

	for file in files:
		(X, y, _, _) = get_img_measurements(B, M, phi, os.path.join(path, file))
		if Xs.size == 0:
			Xs = X
		else:
			Xs = np.concatenate((X, Xs))
		if ys.size == 0:
			ys = y
		else:
			ys = np.concatenate((y, ys))

	return (Xs, ys)


def get_img_measurements(B, M, phi, file):
	image = img.imread(file)
	(szx, szy) = image.shape

	N = B * B
	L = int(szx * szy / (B * B))

	Xs = np.zeros((L , B * B))
	ys = np.zeros((L , M))

	idx = 0

	for i in range(int(szx / B)):
		for j in range(int(szy / B)):
			block = get_block(image, i, j)
			Xs[idx, :] = to_vec((block))
			ys[idx, :] = np.matmul(phi, Xs[idx, :])
			Xs[idx, :] = Xs[idx, :] - np.mean(Xs[idx, :])
			idx += 1

	return (Xs, ys, szx, szy)

def get_img_from_blocks(szx, szy, B, Xs, ys):
	image = np.zeros((szx, szy))
	idx = 0
	for i in range(int(szx / B)):
		for j in range(int(szy / B)):
			image[i * B : (i + 1) * B, j * B : (j + 1) * B] = \
				(to_mat(Xs[idx, :], B)) + ys[idx, -1]
			idx += 1
	return image

def psnr(i1, i2):
	(M, N) = i1.shape
	MSE = (1. / (M * N)) * np.sum((i1 - i2) ** 2)
	return 10 * np.log10(255.0**2 / MSE)  