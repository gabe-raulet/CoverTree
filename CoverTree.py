import time
import numpy as np
import random
import sklearn
from scipy.sparse import csr_array

import cppimport
import cppimport.import_hook
from covertree.ctree import cover_tree_l1, cover_tree_l2, cover_tree_linf, cover_tree_angular, cover_tree_jaccard, cover_tree_hamming

class CoverTree(object):

    def __init__(self, points, metric="euclidean"):
        self.metric = metric
        if self.metric in ("manhattan", "l1"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.tree = cover_tree_l1(self.points)
        elif self.metric in ("euclidean", "l2"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.tree = cover_tree_l2(self.points)
        elif self.metric in ("chebyshev", "infinity"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.tree = cover_tree_linf(self.points)
        elif self.metric in ("angular"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.tree = cover_tree_angular(self.points)
        elif self.metric in ("jaccard"):
            assert points.dtype == np.uint8
            self.convert = lambda p: p
            self.tree = cover_tree_jaccard(self.points)
        elif self.metric in ("hamming"):
            assert type(points) == list and type(points[0] == str)
            d = len(points[0])
            assert set([len(s) for s in points]).pop() == d
            self.convert = lambda p : np.frombuffer("".join(p).encode("ascii"), dtype=np.uint8)
            self.points = np.frombuffer("".join(points).encode("ascii"), dtype=np.uint8).reshape(len(points),-1)
            self.tree = cover_tree_hamming(self.points)

    def build_index(self, cover=1.3, leaf_size=40, num_threads=1):
        self.tree.build_index(cover, leaf_size, num_threads)

    def radius_query(self, query, radius, return_distance=False):
        dists, neighs = self.tree.radius_query(self.convert(query), radius)
        if return_distance: return dists, neighs
        else: return neighs

    def radius_neighbors_graph(self, radius, num_threads=1):
        dists, colids, rowptrs = self.tree.radius_neighbors_graph(radius, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))

def random_word(d):
    return "".join([random.choice("ACTG") for i in range(d)])

n, d = 10000, 31
points = []
for i in range(n):
    points.append(random_word(d))

#  points = np.random.uniform(-1,1, size=(n,d)).astype(np.float32)

tree = CoverTree(points, metric="hamming")
tree.build_index()

