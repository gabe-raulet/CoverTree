import time
import numpy as np
import sklearn
from scipy.sparse import csr_array

import cppimport
import cppimport.import_hook
from extensions.brute_force import brute_force

class BruteForce(object):

    def __init__(self, points, metric):
        self.metric = metric
        self.points = points
        self.bf = brute_force(self.points, self.metric)

    def radius_neighbors_graph(self, radius, num_threads=4):
        dists, colids, rowptrs = self.bf.radius_neighbors_graph(self.points, radius, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))
