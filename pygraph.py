#!/usr/bin/env python3

from mpi4py import MPI
from dataset_io import *
import numpy as np
import math
import sys
#  import getopt
#  import time
#  import json
#  from datetime import datetime
#  from scipy.io import mmwrite
#  from scipy.sparse import csr_array
from metricspace import DistPointVector, DistGraph

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

points = DistPointVector("scratch/datasets/corel.fvecs", comm)
#  points.cover_tree_systolic(0.1, 1.5, 10, 2)
points.cover_tree_voronoi(0.1, 1.5, 10, 25*nprocs, "multiway", "static", -1, 1)
