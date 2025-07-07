#!/usr/bin/env python3

from dataset_io import *

import numpy as np
import math
import sys
import getopt
import time
from datetime import datetime
from scipy.io import mmwrite
from scipy.sparse import csr_array
from bindings.metricspace import *
from sklearn.neighbors import NearestNeighbors

points = read_file("scratch/datasets/deep.fbin", start=0, count=100)
n, d = points.shape

metric = EuclideanSpaceFloat(points)
bf = BruteForceEuclideanSpaceFloat(metric)

nn = NearestNeighbors(algorithm="kd_tree")
nn = nn.fit(points)

g1 = nn.radius_neighbors_graph(points[:10], radius=1.2)
g1 = nn.radius_neighbors_graph(radius=1.2)
g2 = csr_array(bf.radius_neighbors(1.2))

print(g1.__repr__())
print(g2.__repr__())
