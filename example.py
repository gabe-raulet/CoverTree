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
from neighbors import MetricSpace, BruteForce, CoverTree
from sklearn.neighbors import NearestNeighbors

points = read_file("scratch/datasets/corel.fvecs", count=10000)
metric = MetricSpace(points, "euclidean")
tree = CoverTree(metric)
tree.build(cover=1.3, leaf_size=25)
