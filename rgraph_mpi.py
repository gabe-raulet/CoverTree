#!/usr/bin/env python3

from mpi4py import MPI
from dataset_io import *

import numpy as np
import math
import sys
import getopt
import time
import json
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
pack=False
stats=None
cover=1.3
leaf_size=40

def usage():
    global start, count, outfile, metric, method, pack, cover, leaf_size, stats
    if myrank == 0:
        sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <points> -r <radius>\n")
        sys.stderr.write(f"Options: -m STR   metric name [{metric}] (valid: euclidean, manhattan, chebyshev)\n")
        sys.stderr.write(f"         -A STR   method name [{method}] (valid: bruteforce, covertree)\n")
        sys.stderr.write(f"         -n INT   number of points [{'all' if not count else str(count)}]\n")
        sys.stderr.write(f"         -s INT   start offset [{start}]\n")
        sys.stderr.write(f"         -o FILE  output sparse graph [{outfile}]\n")
        sys.stderr.write(f"         -c FLOAT cover tree base [{cover:.3f}]\n")
        sys.stderr.write(f"         -l INT   leaf size [{leaf_size}]\n")
        sys.stderr.write(f"         -j FILE  output stats json\n")
        sys.stderr.write(f"         -p       pack cover tree\n");
        sys.stderr.write(f"         -h       help message\n")
        sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    radius = -1

    try: opts, args = getopt.getopt(sys.argv[1:], "i:r:m:A:n:s:o:l:c:j:ph")
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
        elif o == "-p": pack = True
        elif o == "-j": stats = a
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

    stats_dict = {}
    stats_dict["sizes"] = comm.gather(mysize, root=0)
    stats_dict["filename"] = infile
    stats_dict["datetime"] = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')
    stats_dict["method"] = method
    stats_dict["metric"] = metric
    stats_dict["command"] = " ".join(sys.argv)
    stats_dict["num_points"] = int(n)
    stats_dict["dimension"] = int(d)
    stats_dict["type"] = kind
    stats_dict["offsets"] = comm.gather(int(myoffset), root=0)
    stats_dict["nprocs"] = nprocs

    space = MetricSpace(mypoints, metric)

    if method == "bruteforce":

        t = -MPI.Wtime()
        bf = BruteForce(space)
        mydists, myneighs, myptrs = bf.radius_neighbors_dist(radius, comm)
        t += MPI.Wtime()

        mygraphtime = t

        stats_dict["buildtimes"] = [0]*nprocs
        stats_dict["graphtimes"] = comm.gather(mygraphtime, root=0)

    elif method == "covertree":

        stats_dict["pack"] = pack

        t = -MPI.Wtime()
        tree = CoverTree(space)
        tree.build(cover=cover, leaf_size=leaf_size)
        if pack: tree = tree.get_packed()
        t += MPI.Wtime()

        mybuildtime = t

        myverts = tree.num_vertices()
        mymaxlevel = tree.max_level()

        stats_dict["verts"] = comm.gather(myverts, root=0)
        stats_dict["maxlevels"] = comm.gather(mymaxlevel, root=0)
        stats_dict["buildtimes"] = comm.gather(mybuildtime, root=0)
        stats_dict["cover"] = cover
        stats_dict["leaf_size"] = leaf_size

        if myrank == 0:
            totverts = sum(stats_dict["verts"])
            maxlevel = max(stats_dict["maxlevels"])
            sys.stdout.write(f"[time={t:.3f}] built cover tree [vertices={totverts},maxlevel={maxlevel},cover={cover:.3f},leaf_size={leaf_size}]\n")
            sys.stdout.flush()

        t = -MPI.Wtime()
        mydists, myneighs, myptrs = tree.radius_neighbors_dist(radius, comm)
        t += MPI.Wtime()

        mygraphtime = t
        stats_dict["graphtimes"] = comm.gather(mygraphtime, root=0)

    mynz = len(myneighs)
    stats_dict["edges"] = comm.gather(mynz, root=0)

    if myrank == 0:
        nz = sum(stats_dict["edges"])
        sys.stdout.write(f"[time={t:.3f}] built near neighbor graph [edges={nz},density={nz/n:.3f},method={method},metric={metric}]\n")
        sys.stdout.flush()

        if stats:
            with open(stats, "w") as f:
                json.dump(stats_dict, f, indent=4)

    if outfile:

        comment = f"datetime: {stats_dict['datetime']}\n"
        comment += f"command: '{stats_dict['command']}'\n"
        comment += f"metric: '{metric}'\n"
        comment += f"method: '{method}'\n"
        comment += f"infile: '{infile}'\n"
        comment += f"num: {n}\n"
        comment += f"dim: {d}\n"
        comment += f"type: {kind}\n"
        comment += f"start: {start}\n"
        comment += f"radius: {radius:.4f}\n"
        comment += f"myoffset: {myoffset}\n"
        comment += f"myrank: {myrank}\n"
        comment += f"nprocs: {nprocs}\n"
        comment += f"mybuildtime: {mybuildtime:.4f} seconds\n"
        comment += f"mygraphtime: {mygraphtime:.4f} seconds\n"

        t = -MPI.Wtime()
        mygraph = csr_array((mydists, myneighs, myptrs), shape=(mysize,n))
        mmwrite(f"{outfile}.rank{myrank+1}", mygraph.sorted_indices(), comment=comment, field="pattern", symmetry="symmetric")
        t += MPI.Wtime()

        t = comm.reduce(t, op=MPI.MAX, root=0)

        if myrank == 0:
            sys.stdout.write(f"[time={t:.3f}] wrote graph to file '{outfile}.rank[..]'\n")
            sys.stdout.flush()

