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

points = read_file("scratch/datasets/deep.fbin")
n, d = points.shape

metric = EuclideanSpaceFloat(points)
bf = BruteForceEuclideanSpaceFloat(metric)


