import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy import fftpack

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


def get_img_dataset(szx, szy, B, M, phi, path):
	files = os.listdir(path)
	N = B * B
	L = int(szx * szy / (B * B))

	Xs = np.zeros((L * len(files), B * B))
	ys = np.zeros((L * len(files), M))

	idx = 0

	for file in files:
		image = img.imread(os.path.join(path,file))
		for i in range(int(szx / B)):
			for j in range(int(szy / B)):
				Xs[idx, :] = to_vec(dct2(get_block(image, i, j)))
				ys[idx, :] = np.matmul(phi, Xs[i, :])
				idx += 1

	return (Xs, ys)


def get_img_measurements(szx, szy, B, M, phi, file):
	N = B * B
	L = int(szx * szy / (B * B))

	Xs = np.zeros((L , B * B))
	ys = np.zeros((L , M))

	idx = 0

	image = img.imread(file)
	for i in range(int(szx / B)):
		for j in range(int(szy / B)):
			Xs[idx, :] = to_vec(dct2(get_block(image, i, j)))
			ys[idx, :] = np.matmul(phi, Xs[idx, :])
			idx += 1

	return (Xs, ys)

def get_img_from_dct_blocks(szx, szy, B, Xs):
	image = np.zeros((szx, szy))
	idx = 0
	for i in range(int(szx / B)):
		for j in range(int(szy /B)):
			image[i * B : (i + 1) * B, j * B : (j + 1) * B] = \
				idct2(to_mat(Xs[idx, :], B))
			idx += 1
	return image