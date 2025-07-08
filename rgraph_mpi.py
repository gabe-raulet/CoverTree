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
from neighbors import MetricSpace, BruteForce, CoverTree

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

start=0
count=None
outfile=None
metric="euclidean"
method="bruteforce"
verbose=False
cover=1.3
leaf_size=40

def usage():
    global start, count, outfile, metric, method, verbose, cover, leaf_size
    if myrank == 0:
        sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <points> -r <radius>\n")
        sys.stderr.write(f"Options: -m STR   metric name [{metric}] (valid: euclidean, manhattan, chebyshev)\n")
        sys.stderr.write(f"         -A STR   method name [{method}] (valid: bruteforce, covertree)\n")
        sys.stderr.write(f"         -n INT   number of points [{'all' if not count else str(count)}]\n")
        sys.stderr.write(f"         -s INT   start offset [{start}]\n")
        sys.stderr.write(f"         -o FILE  output sparse graph [{outfile}]\n")
        sys.stderr.write(f"         -c FLOAT cover tree base [{cover:.3f}]\n")
        sys.stderr.write(f"         -l INT   leaf size [{leaf_size}]\n")
        sys.stderr.write(f"         -v       verbose\n");
        sys.stderr.write(f"         -h       help message\n")
        sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    radius = -1

    try: opts, args = getopt.getopt(sys.argv[1:], "i:r:m:A:n:s:o:l:c:vh")
    except getopt.GetoptError as err: usage()

    for o, a in opts:
        if o == "-i": infile = a
        elif o == "-r": radius = float(a)
        elif o == "-m": metric = a
        elif o == "-A": method = a
        elif o == "-n": count = int(a)
        elif o == "-s": start = int(a)
        elif o == "-o": outfile = a
        elif o == "-c": cover = float(a)
        elif o == "-l": leaf_size = int(a)
        elif o == "-v": verbose = True
        elif o == "-h": usage()
        else: assert False, "unhandled option"

    if infile is None or radius < 0: usage()

    t = -MPI.Wtime()
    mypoints, myoffset, n, d, kind = read_file_dist(infile, comm, start=start, count=count)
    t += MPI.Wtime()

    t = comm.reduce(t, op=MPI.MAX, root=0)
    mysize = mypoints.shape[0]

    if myrank == 0:
        sys.stdout.write(f"[time={t:.3f}] read points[{start}..{start+n}) from '{infile}' [n={n},d={d},type={kind}]\n")
        sys.stdout.flush()

    if verbose:
        allsizes = comm.gather(mysize, root=0)
        if myrank == 0:
            for rank in range(nprocs):
                sys.stdout.write(f"[rank={rank},mysize={allsizes[rank]}]\n")
                sys.stdout.flush()

    space = MetricSpace(mypoints, metric)

    if method == "bruteforce":

        t = -MPI.Wtime()
        bf = BruteForce(space)
        mydists, myneighs, myptrs = bf.radius_neighbors_dist(radius, comm)
        t += MPI.Wtime()

        mynz = len(myneighs)

        t = comm.reduce(t, op=MPI.MAX, root=0)
        nz = comm.reduce(mynz, op=MPI.SUM)

    elif method == "covertree":

        t = -MPI.Wtime()
        tree = CoverTree(space)
        tree.build(cover=cover, leaf_size=leaf_size)
        t += MPI.Wtime()

        myverts = tree.num_vertices()
        mymaxlevel = tree.max_level()

        t = comm.reduce(t, op=MPI.MAX, root=0)
        verts = comm.reduce(myverts, op=MPI.SUM, root=0)
        maxlevel = comm.reduce(mymaxlevel, op=MPI.MAX, root=0)

        if myrank == 0:
            sys.stdout.write(f"[time={t:.3f}] built cover tree [vertices={verts},maxlevel={maxlevel}]\n")
            sys.stdout.flush()

        if verbose:
            allverts = comm.gather(myverts, root=0)
            allmaxlevels = comm.gather(mymaxlevel, root=0)
            if myrank == 0:
                for rank in range(nprocs):
                    sys.stdout.write(f"[rank={rank},myverts={allverts[rank]},mymaxlevel={allmaxlevels[rank]}]\n")
                    sys.stdout.flush()

        t = -MPI.Wtime()
        mydists, myneighs, myptrs = tree.radius_neighbors_dist(radius, comm)
        t += MPI.Wtime()

        mynz = len(myneighs)

        t = comm.reduce(t, op=MPI.MAX, root=0)
        nz = comm.reduce(mynz, op=MPI.SUM)

    if myrank == 0:
        sys.stdout.write(f"[time={t:.3f}] built near neighbor graph [edges={nz},density={nz/n:.3f},method={method}]\n")
        sys.stdout.flush()

    if verbose:
        alledges = comm.gather(mynz, root=0)
        if myrank == 0:
            for rank in range(nprocs):
                sys.stdout.write(f"[rank={rank},myedges={alledges[rank]},mydensity={alledges[rank]/allsizes[rank]:.3f}]\n")
                sys.stdout.flush()
