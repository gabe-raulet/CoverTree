#!/usr/bin/env python3

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
from sklearn.neighbors import NearestNeighbors

start=0
count=None
outfile=None
num_threads=1
metric="euclidean"
method="bruteforce"
cover=1.3
leaf_size=40

def usage():
    global start, count, outfile, num_threads, metric, method, cover, leaf_size
    sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <points> -r <radius>\n")
    sys.stderr.write(f"Options: -m STR   metric name [{metric}] (valid: euclidean, manhattan, chebyshev)\n")
    sys.stderr.write(f"         -A STR   method name [{method}] (valid: bruteforce, covertree, scikit)\n")
    sys.stderr.write(f"         -n INT   number of points [{'all' if not count else str(count)}]\n")
    sys.stderr.write(f"         -s INT   start offset [{start}]\n")
    sys.stderr.write(f"         -t INT   number of threads [{num_threads}]\n")
    sys.stderr.write(f"         -o FILE  output sparse graph [{outfile}]\n")
    sys.stderr.write(f"         -c FLOAT cover tree base [{cover:.3f}]\n")
    sys.stderr.write(f"         -l INT   leaf size [{leaf_size}]\n")
    sys.stderr.write(f"         -h       help message\n")
    sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    radius = -1

    try: opts, args = getopt.getopt(sys.argv[1:], "i:r:m:A:n:s:t:o:l:c:h")
    except getopt.GetoptError as err: usage()

    for o, a in opts:
        if o == "-i": infile = a
        elif o == "-r": radius = float(a)
        elif o == "-m": metric = a
        elif o == "-A": method = a
        elif o == "-n": count = int(a)
        elif o == "-s": start = int(a)
        elif o == "-t": num_threads = int(a)
        elif o == "-o": outfile = a
        elif o == "-c": cover = float(a)
        elif o == "-l": leaf_size = int(a)
        elif o == "-h": usage()
        else: assert False, "unhandled option"

    if infile is None or radius < 0: usage()

    t = -time.perf_counter()
    n, d, filesize, kind = info_file(infile)
    points = read_file(infile, start, count)
    t += time.perf_counter()

    n, d = points.shape

    sys.stdout.write(f"[time={t:.3f}] read points[{start}..{start+n}) from '{infile}' [n={n},d={d},type={kind}]\n")
    sys.stdout.flush()

    space = MetricSpace(points, metric)

    if method == "bruteforce":

        t = -time.perf_counter()
        bf = BruteForce(space)
        dists, neighs, ptrs = bf.radius_neighbors(radius=radius, return_distance=True, num_threads=num_threads)
        t += time.perf_counter()

        nz = len(neighs)
        runtime = t

    elif method == "covertree":

        t = -time.perf_counter()
        tree = CoverTree(space)
        tree.build(cover=cover, leaf_size=leaf_size)
        t += time.perf_counter()

        sys.stdout.write(f"[time={t:.3f}] built cover tree [vertices={tree.num_vertices()},maxlevel={tree.max_level()}]\n")
        sys.stdout.flush()

        t = -time.perf_counter()
        dists, neighs, ptrs = tree.radius_neighbors(radius=radius, return_distance=True, num_threads=num_threads)
        t += time.perf_counter()

        nz = len(neighs)
        runtime = t

    elif method == "scikit":

        t = -time.perf_counter()
        nn = NearestNeighbors(algorithm="ball_tree").fit(points)
        graph = nn.radius_neighbors_graph(X=points, radius=radius, mode="distance")
        dists, neighs, ptrs = graph.data, graph.indices, graph.indptr
        t += time.perf_counter()

        nz = len(neighs)
        runtime = t

    else:
        raise Exception("Method not implemented")

    sys.stdout.write(f"[time={runtime:.3f}] built near neighbor graph [edges={nz},density={nz/n:.3f},method={method}]\n")
    sys.stdout.flush()

    if outfile:

        cmd = " ".join(sys.argv)
        comment = f"datetime: {datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')}\n"
        comment += f"command: '{cmd}'\n"
        comment += f"metric: '{metric}'\n"
        comment += f"method: '{method}'\n"
        comment += f"infile: '{infile}'\n"
        comment += f"num: {n}\n"
        comment += f"dim: {d}\n"
        comment += f"type: {kind}\n"
        comment += f"start: {start}\n"
        comment += f"radius: {radius:.4f}\n"
        comment += f"threads: {num_threads}\n"
        comment += f"time: {runtime:.4f} seconds\n"

        t = -time.perf_counter()
        graph = csr_array((dists, neighs, ptrs), shape=(n,n))
        mmwrite(outfile, graph.sorted_indices(), comment=comment, field="pattern", symmetry="symmetric")
        t += time.perf_counter()

        sys.stdout.write(f"[time={t:.3f}] wrote graph to file '{outfile}'\n")
        sys.stdout.flush()

    sys.exit(0)
