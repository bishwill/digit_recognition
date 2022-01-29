import numpy as np


def pca(X, dim):
    X = X - np.sum(X, axis=0) / X.shape[0]
    U, S, Vt = np.linalg.svd(X)
    return np.matmul(U[:, :dim], np.diag(S)[:dim, :dim])


if __name__ == '__main__':    
    pass
    