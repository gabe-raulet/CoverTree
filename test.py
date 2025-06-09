import time
import numpy as np
import sklearn
from sklearn.neighbors import KDTree, BallTree, radius_neighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_array
from scipy.io import mmwrite, mmread
from BruteForce import BruteForce

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
        assert metric in ("euclidean", "cosine", "manhattan", 'l1', "l2")
        assert method in ("balltree", "kdtree", "covertree", "bruteforce")
        self.points = points
        self.method = method
        self.metric = metric

    def build_index(self, **kwargs):
        if self.method == "balltree":
            self.index = BallTree(self.points, metric=self.metric, **kwargs)
        elif self.method == "kdtree":
            self.index = KDTree(self.points, metric=self.metric, **kwargs)
        elif self.method == "bruteforce":
            self.index = BruteForce(self.points, metric=self.metric, **kwargs)

    def radius_neighbors_graph(self, radius, num_threads=1):
        if self.method == "balltree":
            return csr_nng(self.index.query_radius(self.points, return_distance=True, r=radius))
        elif self.method == "kdtree":
            return csr_nng(self.index.query_radius(self.points, return_distance=True, r=radius))
        elif self.method == "bruteforce":
            return self.index.radius_neighbors_graph(radius, num_threads)


n, d = 1000, 16
points = np.random.uniform(0,1, size=(n,d))

rng_balltree = RadiusNeighborsGraph(points, "balltree", "euclidean")
rng_balltree.build_index()

rng_kdtree = RadiusNeighborsGraph(points, "kdtree", "euclidean")
rng_kdtree.build_index()

rng_bruteforce = RadiusNeighborsGraph(points, "bruteforce", "euclidean")
rng_bruteforce.build_index()

neighs_balltree = rng_balltree.radius_neighbors_graph(1.1)
neighs_kdtree = rng_kdtree.radius_neighbors_graph(1.1)
neighs_bruteforce = rng_bruteforce.radius_neighbors_graph(1.1)

print(np.allclose(neighs_balltree.todense(), neighs_kdtree.todense()))
print(np.allclose(neighs_balltree.todense(), neighs_bruteforce.todense()))
print(np.allclose(neighs_kdtree.todense(), neighs_bruteforce.todense()))
