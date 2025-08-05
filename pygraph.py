#!/usr/bin/env python3

from mpi4py import MPI
from dataset_io import *
import numpy as np
import math
import sys
from metricspace import DistPointVector, DistGraph

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

points = DistPointVector("scratch/datasets/corel.fvecs", comm)

comm.barrier()

t = -MPI.Wtime()
graph = points.cover_tree_voronoi(0.1, 1.5, 10, 25*nprocs, "multiway", "steal", -1, 2)
t += MPI.Wtime()
maxtime = comm.reduce(t, op=MPI.MAX, root=0)

if myrank == 0:
    print(f"Finished in {maxtime:.3f} seconds")



graph.write_edge_file(points.totsize(), "output.mtx")
