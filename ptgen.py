import os
import sys
import numpy as np
import math
import time
from dataset_io import *
from sklearn.datasets import make_blobs

def random_rotation_matrix(d):
    A = np.random.normal(0, 1, (d, d))
    Q, R = np.linalg.qr(A)

    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q

def main(n, dim, ambient, centers, fname):
    p1 = make_blobs(n_samples=n, n_features=dim, centers=centers)[0]
    p2 = np.zeros((n, ambient-dim))
    A = np.hstack([p1,p2])
    R = random_rotation_matrix(ambient)
    points = (A@R).astype(np.float32)
    write_file(fname, points)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <n> <dim> <ambient> <centers> <fname>")
        sys.exit(1)
    else:
        sys.exit(main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]))
