import numpy as np

def get_measurement_matrix_with_embedded_average(M, N, seed):
	np.random.seed(seed)
	phi = np.random.randn(M, N)
	phi[M - 1, :] = 1.0 / N
	return phi
