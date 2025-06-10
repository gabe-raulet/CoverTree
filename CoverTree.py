import time
import numpy as np
import math
import random
import sklearn
from collections import namedtuple
from scipy.sparse import csr_array
from heapq import heappush, heappop

import cppimport
import cppimport.import_hook
from covertree.ctree import cover_tree_l1, cover_tree_l2, cover_tree_linf, cover_tree_angular, cover_tree_jaccard, cover_tree_hamming

def manhattan_distance(p, q):
    return sum([abs(x-y) for x,y in zip(p,q)])

def euclidean_distance(p, q):
    return np.linalg.norm(p-q)

def angular_distance(p, q):
    pq = np.dot(p,q)
    pp = np.dot(p,p)
    qq = np.dot(q,q)
    if pp*qq == 0: return 0
    val = pq / math.sqrt(pp*qq)
    return math.acos(val) / math.pi

def chebyshev_distance(p, q):
    return max([abs(x-y) for x,y in zip(p,q)])

def jaccard_distance(p, q):
    both = sum([1 for x,y in zip(p,q) if x == y and x != 0])
    either = sum([1 for x,y in zip(p,q) if x != y])
    if either == 0: return 0
    return 1 - (both/either)

def hamming_distance(p, q):
    return sum([1 for x,y in zip(p,q) if x != y]) / len(p)

Vertex = namedtuple("Vertex", ["vertex", "center", "level", "radius", "children", "leaves"])

class CoverTree(object):

    def __init__(self, points, metric="euclidean"):
        self.metric = metric
        if self.metric in ("manhattan", "l1"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.pydist = manhattan_distance
            self.tree = cover_tree_l1(self.points)
        elif self.metric in ("euclidean", "l2"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.pydist = euclidean_distance
            self.tree = cover_tree_l2(self.points)
        elif self.metric in ("chebyshev", "infinity"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.pydist = chebyshev_distance
            self.tree = cover_tree_linf(self.points)
        elif self.metric in ("angular"):
            assert points.dtype == np.float32
            self.points = points
            self.convert = lambda p: p
            self.pydist = angular_distance
            self.tree = cover_tree_angular(self.points)
        elif self.metric in ("jaccard"):
            assert points.dtype == np.uint8
            self.convert = lambda p: p
            self.pydist = jaccard_distance
            self.tree = cover_tree_jaccard(self.points)
        elif self.metric in ("hamming"):
            assert type(points) == list and type(points[0] == str)
            d = len(points[0])
            assert set([len(s) for s in points]).pop() == d
            self.convert = lambda p : np.frombuffer("".join(p).encode("ascii"), dtype=np.uint8)
            self.pydist = hamming_distance
            self.points = np.frombuffer("".join(points).encode("ascii"), dtype=np.uint8).reshape(len(points),-1)
            self.tree = cover_tree_hamming(self.points)

    def build_index(self, cover=1.3, leaf_size=40, num_threads=1):
        self.tree.build_index(cover, leaf_size, num_threads)
        self.cover = cover

    def radius_query(self, query, radius, return_distance=False):
        dists, neighs = self.tree.radius_query(self.convert(query), radius)
        if return_distance: return dists, neighs
        else: return neighs

    def knn_query(self, query, k=10, return_distance=False):
        dists, neighs = self.tree.knn_query(self.convert(query), k)
        if return_distance: return dists, neighs
        else: return neighs

    def radius_neighbors_graph(self, radius, num_threads=1):
        dists, colids, rowptrs = self.tree.radius_neighbors_graph(radius, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))

    def kneighbors_graph(self, k=10, num_threads=1):
        dists, colids, rowptrs = self.tree.kneighbors_graph(k, num_threads)
        return csr_array((dists, colids, rowptrs), shape=(len(rowptrs)-1, len(rowptrs)-1))

    def get_vertex(self, vertex):
        return Vertex._make(self.tree.get_vertex(vertex))

    def bfs(self, root):
        queue = [self.get_vertex(root)]
        while queue:
            v = queue.pop(0)
            yield v
            for child in v.children:
                queue.append(self.get_vertex(child))

    def dfs(self, root):
        stack = [self.get_vertex(root)]
        while stack:
            v = stack.pop()
            yield v
            for child in reversed(v.children):
                stack.append(self.get_vertex(child))

    def reorder(self):
        ordering = np.array([v.vertex for v in self.dfs(0)])
        self.tree.reorder_vertices(ordering)

    def __getitem__(self, vertex):
        return self.get_vertex(vertex)

#  n, d = 10000, 8

#  points = np.random.uniform(-1,1, size=(n,d)).astype(np.float32)
#  tree.build_index(cover=1.3, leaf_size=10)

n, d = 15, 3
np.random.seed(103)
points = np.random.uniform(-1,1, size=(n,d)).astype(np.float32)

tree = CoverTree(points, metric="euclidean")
tree.build_index(cover=1.3, leaf_size=5)

#  from sklearn.neighbors import kneighbors_graph
#  from scipy.spatial import KDTree

#  kdtree = KDTree(points)

#  graph1 = kneighbors_graph(points,10, metric="euclidean", include_self = True)
#  graph2 = tree.kneighbors_graph(k=10, num_threads=12)

#  graph1.sort_indices()
#  graph2.sort_indices()

#  assert np.all(graph1.indices == graph2.indices)
#  assert np.all(graph1.indptr == graph2.indptr)
