#!/usr/bin/env python3

#  from mpi4py import MPI
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

def MetricSpace(points, metric="euclidean"):
    if metric == "euclidean":
        if points.dtype == "float32": return EuclideanSpaceFloat(points)
        elif points.dtype == "float64": return EuclideanSpaceDouble(points)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if points.dtype == "float32": return ChebyshevSpaceFloat(points)
        elif points.dtype == "float64": return ChebyshevSpaceDouble(points)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

points = read_file("scratch/datasets/corel.fvecs", count=1000)

metric = MetricSpace(points.astype("float32"), "chebyshev")

#  comm = MPI.COMM_WORLD
#  myrank = comm.Get_rank()
#  nprocs = comm.Get_size()

#  mypoints, myoffset, totsize, d, kind = read_file_dist("scratch/datasets/corel.fvecs", comm, start=0, count=None)

#  metric = EuclideanSpaceFloat(mypoints)
#  bf = BruteForceEuclideanSpaceFloat(metric)

#  t = -time.perf_counter()
#  g = csr_array(bf.radius_neighbors(0.15, return_distance=True, num_threads=12))
#  t += time.perf_counter()

#  tot=mypoints.shape[0]
#  print(f"[myrank={myrank},mysize={metric.num_points()},myedges={g.nnz},totsize={tot},density={g.nnz/tot:.3f}] runtime={t:.3f}")

#  t = -time.perf_counter()
#  tree = CoverTreeEuclideanSpaceFloat(metric)
#  tree.build(cover=1.3, leaf_size=40)
#  g = csr_array(tree.radius_neighbors(0.15, return_distance=True, num_threads=12))
#  t += time.perf_counter()

#  tot=mypoints.shape[0]
#  print(f"[myrank={myrank},mysize={metric.num_points()},myedges={g.nnz},totsize={tot},density={g.nnz/tot:.3f}] runtime={t:.3f}")
