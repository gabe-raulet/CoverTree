import time
import numpy as np
import sklearn
from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics.pairwise import euclidean_distances

import cppimport
import cppimport.import_hook
from extensions.brute_force import distance_matrix

class BruteForce(object):

    def __init__(self, points):
        self.points = points
        self.dists = distance_matrix(points, num_threads=8)

n = 20000
d = 12

points = np.random.uniform(0,1,size=(n,d))

brute_force = BruteForce(points)
tree = KDTree(points)


