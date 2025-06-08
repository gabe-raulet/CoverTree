import time
import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances

import cppimport
import cppimport.import_hook
from extensions.brute_force import distance_matrix

n = 20000
d = 32

np.random.seed(134)
points = np.random.uniform(0,1,size=(n,d))

num_threads=12

t = -time.perf_counter()
D = distance_matrix(points, num_threads=num_threads)
t += time.perf_counter()

