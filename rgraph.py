#!/usr/bin/env python3

from mpi4py import MPI
from dataset_io import *

import numpy as np
import math
import sys
import getopt
import json
from datetime import datetime
from metricspace import DistPointVector, DistGraph

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

outfile=None
method="cvor" # or gvor or bf or ct
cover=1.55
leaf_size=10
num_centers=25
verbosity=1

tree_assignment="multiway"
query_balancing="static"
queries_per_tree=-1

def usage():
    global outfile, method, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree
    if myrank == 0:
        sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <points> -r <radius>\n")
        sys.stderr.write(f"Options: -c FLOAT cover tree base [{cover:.3f}]\n")
        sys.stderr.write(f"         -l INT   leaf size [{leaf_size}]\n")
        sys.stderr.write(f"         -m INT   num centers [{num_centers}]\n")
        sys.stderr.write(f"         -q INT   queries per tree [{queries_per_tree}]\n")
        sys.stderr.write(f"         -v INT   verbosity level [{verbosity}]\n")
        sys.stderr.write(f"         -o FILE  output sparse graph\n")
        sys.stderr.write(f"         -A STR   tree assignment method (one of: static, multiway) [{tree_assignment}]\n")
        sys.stderr.write(f"         -B STR   load balancing method (one of: static, steal) [{query_balancing}]\n")
        sys.stderr.write(f"         -M STR   querying method (one of: gvor, cvor, ct, bf) [{method}]\n")
        sys.stderr.write(f"         -F       fix total centers\n")
        sys.stderr.write(f"         -h       help message\n")
        sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    radius = -1

    fix_num_centers = False

    try: opts, args = getopt.getopt(sys.argv[1:], "c:l:m:Fv:o:i:r:q:A:B:M:h")
    except getopt.GetoptError as err: usage()

    for o, a in opts:
        if o == "-i": infile = a
        elif o == "-r": radius = float(a)
        elif o == "-c": cover = float(a)
        elif o == "-l": leaf_size = int(a)
        elif o == "-m": num_centers = int(a)
        elif o == "-q": queries_per_tree = int(a)
        elif o == "-v": verbosity = int(a)
        elif o == "-o": outfile = a
        elif o == "-A": tree_assignment = a
        elif o == "-B": query_balancing = a
        elif o == "-F": fix_num_centers = True
        elif o == "-M": method = a
        elif o == "-h": usage()
        else: assert False, "unhandled option"

    if not fix_num_centers: num_centers *= nprocs

    if infile is None or radius < 0: usage()

    if not tree_assignment in ("static", "multiway"):
        sys.stderr.write(f"error: '{tree_assignment}' is an invalid tree assignment method!\n")
        sys.stdout.flush()
        sys.exit(1)

    if not query_balancing in ("static", "steal"):
        sys.stderr.write(f"error: '{query_balancing}' is an invalid query balancing method!\n")
        sys.stdout.flush()
        sys.exit(1)

    if not method in ("ct", "bf", "gvor", "cvor"):
        sys.stderr.write(f"error: '{method}' is an invalid query method!\n")
        sys.stdout.flush()
        sys.exit(1)

    comm.barrier()
    t = -MPI.Wtime()
    points = DistPointVector(infile, comm)
    t += MPI.Wtime()
    maxtime = comm.reduce(t, op=MPI.MAX, root=0)

    num_vertices = points.totsize()

    if verbosity >= 1:
        if myrank == 0: sys.stdout.write(f"[v1,time={maxtime:.3f}] Read file '{infile}' [size={num_vertices},dim={points.num_dimensions()}]\n")
        sys.stdout.flush()

    t = -MPI.Wtime()

    if method == "bf": graph = points.brute_force_systolic(radius, verbosity)
    elif method == "ct": graph = points.cover_tree_systolic(radius, cover, leaf_size, verbosity)
    elif method == "gvor": graph = points.ghost_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree, verbosity)
    elif method == "cvor": graph = points.cover_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree, verbosity)

    t += MPI.Wtime()
    maxtime = comm.reduce(t, op=MPI.MAX, root=0)

    dist_comps = comm.gather(points.dist_comps(), root=0)
    comm_times = comm.gather(points.my_comm_time(), root=0)
    comp_times = comm.gather(points.my_comp_time(), root=0)
    idle_times = comm.gather(points.my_idle_time(), root=0)

    if myrank == 0:
        for i in range(nprocs):
            sys.stdout.write(f"[rank={i},dist_comps={format_large_number(dist_comps[i])},comm_time={comm_times[i]:.3f},comp_time={comp_times[i]:.3f},idle_time={idle_times[i]:.3f}]\n")
        sys.stdout.write(f"[dist_comps={format_large_number(sum(dist_comps))}]\n")

    sys.stdout.flush()
    comm.barrier()

    edges = graph.num_edges(num_vertices,0)
    density = edges/num_vertices

    if myrank == 0: sys.stdout.write(f"[v0,time={maxtime:.3f}] found neighbors [vertices={num_vertices},edges={edges},density={density:.3f}]\n")

    sys.stdout.flush()


    if outfile:

        t = -MPI.Wtime()
        graph.write_edge_file(num_vertices, outfile)
        t += MPI.Wtime()
        maxtime = comm.reduce(t, op=MPI.MAX, root=0)

        if verbosity >= 1:
            if myrank == 0:
                sys.stdout.write(f"[v1,time={maxtime:.3f}] wrote edges to file '{outfile}'\n")
            sys.stdout.flush()

    sys.exit(0)
