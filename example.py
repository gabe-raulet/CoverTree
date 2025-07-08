#!/usr/bin/env python3

from dataset_io import *
from mpi4py import MPI

import numpy as np
import math
import sys
import getopt
import time
from datetime import datetime
from scipy.io import mmwrite
from scipy.sparse import csr_array
from bindings.metricspace import *
from neighbors import MetricSpace, BruteForce, CoverTree

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

mypoints, myoffset, totsize, d, kind = read_file_dist("scratch/datasets/corel.fvecs", comm)

#  print(f"[myrank={myrank},mysize={len(mypoints)},myoffset={myoffset},totsize={totsize},d={d},kind={kind}]")

metric = MetricSpace(mypoints, "euclidean")
bf = BruteForce(metric)

mydists, myneighs, myptrs = bf.radius_neighbors_dist(0.125, comm)

nz = len(myneighs)
totnz = comm.allreduce(nz, op=MPI.SUM)

print(f"[myrank={myrank},mysize={len(mypoints)},myoffset={myoffset},totsize={totsize},d={d},kind={kind},myedges={nz},edges={totnz},myptrs={len(myptrs)}]")



#  metric = MetricSpace(points, "euclidean")
#  tree = CoverTree(metric)
#  tree.build(cover=1.3, leaf_size=25)
