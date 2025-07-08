#!/usr/bin/env python3

from mpi4py import MPI
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
from sklearn.neighbors import NearestNeighbors

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

#  points = read_file_dist("scratch/datasets/deep.fbin", start=0, count=4000)
#  n, d = points.shape

mypoints, myoffset, totsize, d, kind = read_file_dist("scratch/datasets/deep.fbin", comm, start=0, count=10000)

#  print(f"[myrank={myrank},mysize={mypoints.shape[0]},myoffset={myoffset},totsize={totsize},dim={d},kind={kind}]")

metric = EuclideanSpaceFloat(mypoints)
bf = BruteForceEuclideanSpaceFloat(metric)

#  g1 = csr_array(bf.radius_neighbors(1.2, comm, return_distance=True))
#  g1 = csr_array(bf.radius_neighbors_dist(1.2, comm, True))

t = -time.perf_counter()
data, colids, indptr = bf.radius_neighbors_dist(1.2, comm, True)
t += time.perf_counter()

maxtime = comm.allreduce(t, op=MPI.MAX)
g1 = csr_array((data, colids, indptr), shape=(len(mypoints), totsize))

nz = g1.nnz
tot = comm.allreduce(nz, op=MPI.SUM)

print(f"[myrank={myrank},mysize={metric.num_points()},myedges={g1.nnz},totsize={tot}] runtime={maxtime:.3f}")

#  metric = EuclideanSpaceFloat(points)
#  bf = BruteForceEuclideanSpaceFloat(metric)

#  nn = NearestNeighbors(algorithm="kd_tree")
#  nn = nn.fit(points)

#  g1 = nn.radius_neighbors_graph(points, radius=1.05)
#  g2 = csr_array(bf.radius_neighbors(1.05, num_threads=1))
#  g3 = csr_array(bf.radius_neighbors(1.05, num_threads=8))


#  print(g1.__repr__())
#  print(g2.__repr__())
