import numpy as np

def make_U_triangular(size):
    # create upper triangle matrix
    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i, size):
            matrix[i, j] = 1
    return matrix

def make_L_triangular(size):
    # create lower triangle matrix
    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i, size):
            matrix[j, i] = 1
    return matrix