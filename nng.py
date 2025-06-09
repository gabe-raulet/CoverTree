#!/usr/bin/env python3

import sys
import getopt
import time
import numpy as np
from scipy.io import mmwrite
from dataset.dataset_io import *
from rng import RadiusNeighborsGraph

start=0
count=None
outfile=None
num_threads=1
metric="euclidean"
method="sklearn_pairwise"

def usage():
    global start, count, outfile, num_threads, metric, method
    sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <points> -r <radius>\n")
    sys.stderr.write(f"Options: -m STR  metric name [{metric}] (valid: euclidean, manhattan, chebyshev)\n")
    sys.stderr.write(f"         -A STR  method name [{method}] (valid: sklearn_pairwise, covertree, bruteforce, balltree, kdtree)\n")
    sys.stderr.write(f"         -n INT  number of points [{'all' if not count else str(count)}]\n")
    sys.stderr.write(f"         -s INT  start offset [{start}]\n")
    sys.stderr.write(f"         -t INT  number of threads [{num_threads}]\n")
    sys.stderr.write(f"         -o FILE output sparse graph [{outfile}]\n")
    sys.stderr.write(f"         -h      help message\n")
    sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    points_fname = None
    radius = -1

    try: opts, args = getopt.getopt(sys.argv[1:], "i:r:m:A:n:s:t:o:h")
    except getopt.GetoptError as err: usage()

    for o, a in opts:
        if o == "-i": points_fname = a
        elif o == "-r": radius = float(a)
        elif o == "-m": metric = a
        elif o == "-A": method = a
        elif o == "-n": count = int(a)
        elif o == "-s": start = int(a)
        elif o == "-t": num_threads = int(a)
        elif o == "-o": outfile = a
        elif o == "-h": usage()
        else: assert False, "unhandled option"

    if points_fname is None or radius < 0: usage()

    t = -time.perf_counter()
    points = read_file(points_fname, start, count)
    t += time.perf_counter()

    n, d = points.shape

    sys.stdout.write(f"[time={t:.3f}] read '{points_fname}' [n={n},d={d}]\n")
    sys.stdout.flush()

    t = -time.perf_counter()
    rng_index = RadiusNeighborsGraph(points, method, metric)
    rng_index.build_index()
    t += time.perf_counter()

    sys.stdout.write(f"[time={t:.3f}] built index [method='{method}']\n")
    sys.stdout.flush()

    t = -time.perf_counter()
    graph = rng_index.radius_neighbors_graph(radius=radius, num_threads=num_threads)
    t += time.perf_counter()

    sys.stdout.write(f"[time={t:.3f}] built near neighbor graph [edges={graph.nnz},density={graph.nnz/n:.3f}]\n")
    sys.stdout.flush()

    if outfile:
        t = -time.perf_counter()
        mmwrite(outfile, graph.sorted_indices(), field="pattern", symmetry="symmetric")
        t += time.perf_counter()
        sys.stdout.write(f"[time={t:.3f}] wrote graph to file '{outfile}'\n")
        sys.stdout.flush()
