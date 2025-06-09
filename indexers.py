import time
import numpy as np
import sklearn
from scipy.sparse import csr_array

import cppimport
import cppimport.import_hook
from extensions.brute_force import brute_force
from extensions.cover_tree import cover_tree_euclidean, cover_tree_manhattan, cover_tree_chebyshev, cover_tree_angular

class BruteForce(object):

    def __init__(self, points, metric):
        self.metric = metric
        self.points = points
        self.bf = brute_force(self.points, self.metric)

    def radius_neighbors_graph(self, radius, num_threads=1):
        dists, colids, rowptrs = self.bf.radius_neighbors_graph(self.points, radius, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))

class CoverTree(object):

    def __init__(self, points, metric, cover=1.3, leaf_size=30, num_threads=1):
        assert metric in ("euclidean", "manhattan", "chebyshev", "angular")
        self.metric = metric
        self.points = points
        if self.metric == "euclidean": self.tree = cover_tree_euclidean(self.points)
        elif self.metric == "manhattan": self.tree = cover_tree_manhattan(self.points)
        elif self.metric == "chebyshev": self.tree = cover_tree_chebyshev(self.points)
        elif self.metric == "angular": self.tree = cover_tree_angular(self.points)
        self.tree.build_index(self.points, cover, leaf_size, num_threads)

    def radius_neighbors_graph(self, radius, num_threads=1):
        dists, colids, rowptrs = self.tree.radius_neighbors_graph(self.points, radius, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))

    def range_query(self, q, radius, return_distance=False):
        dists, neighbors = self.tree.range_query(self.points, self.points[q], radius)
        if return_distance: return dists, neighbors
        else: return neighbors
