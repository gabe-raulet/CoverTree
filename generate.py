import sys
import numpy as np
import os
from dataset_io import *
from sklearn.datasets import make_blobs

points = make_blobs(n_samples=100000, n_features=128, centers=80)[0].astype("float32")
write_file("points.fvecs", points)

