import time
import numpy as np
import sklearn
from itertools import combinations
from sklearn.neighbors import KDTree, BallTree, radius_neighbors_graph
from sklearn.metrics.pairwise import distance_metrics
from scipy.sparse import csr_array
from scipy.io import mmwrite, mmread
from indexers import BruteForce, CoverTree

def csr_nng(neighs):
    n = len(neighs[0])
    rowptrs, colids, dists = [], [], []
    for i in range(n):
        rowptrs.append(len(colids))
        colids += list(neighs[0][i])
        dists += list(neighs[1][i])
    rowptrs.append(len(colids))
    return csr_array((dists, colids, rowptrs), shape=(n,n))

class RadiusNeighborsGraph(object):

    def __init__(self, points, method="balltree", metric="euclidean"):
        assert metric in ("euclidean", "cosine", "manhattan", 'l1', "l2", "chebyshev", "infinity", "angular")
        assert method in ("balltree", "kdtree", "covertree", "bruteforce", "sklearn_pairwise")
        self.points = points
        self.method = method
        self.metric = metric
        if self.method == "balltree":
            assert self.metric in BallTree.valid_metrics
        elif self.method == "kdtree":
            assert self.metric in KDTree.valid_metrics
        elif self.method == "sklearn_pairwise":
            assert self.metric in distance_metrics()

    def build_index(self, num_threads=1, **kwargs):
        if self.method == "balltree":
            self.index = BallTree(self.points, metric=self.metric, **kwargs)
        elif self.method == "kdtree":
            self.index = KDTree(self.points, metric=self.metric, **kwargs)
        elif self.method == "bruteforce":
            self.index = BruteForce(self.points, metric=self.metric, **kwargs)
        elif self.method == "covertree":
            self.index = CoverTree(self.points, metric=self.metric, num_threads=num_threads, **kwargs)
        elif self.method == "sklearn_pairwise":
            pass

    def radius_neighbors_graph(self, radius, num_threads=1):
        if self.method == "balltree":
            return csr_nng(self.index.query_radius(self.points, return_distance=True, r=radius))
        elif self.method == "kdtree":
            return csr_nng(self.index.query_radius(self.points, return_distance=True, r=radius))
        elif self.method == "bruteforce":
            return self.index.radius_neighbors_graph(radius, num_threads)
        elif self.method == "covertree":
            return self.index.radius_neighbors_graph(radius, num_threads)
        elif self.method == "sklearn_pairwise":
            return radius_neighbors_graph(self.points, radius, mode="distance", metric=self.metric, include_self=True, n_jobs=num_threads)
