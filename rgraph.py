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
num_centers=10
verbosity=1
stats = {}
stats_file=None

tree_assignment="multiway"
query_balancing="static"
queries_per_tree=-1

def usage():
    global outfile, method, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree, stats_file
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
        sys.stderr.write(f"         -M STR   querying method (one of: gvor, cvor, ct, bf, ctrma) [{method}]\n")
        sys.stderr.write(f"         -j FILE  stats file\n")
        sys.stderr.write(f"         -F       fix total centers\n")
        sys.stderr.write(f"         -h       help message\n")
        sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    radius = -1

    fix_num_centers = False

    try: opts, args = getopt.getopt(sys.argv[1:], "c:l:m:Fv:o:i:r:q:A:B:M:j:h")
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
        elif o == "-j": stats_file = a
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

    if not method in ("ct", "bf", "gvor", "cvor", "ctrma"):
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

    parameters = {}
    parameters["infile"] = infile
    parameters["method"] = method
    parameters["cover"] = cover
    parameters["leaf_size"] = leaf_size
    parameters["num_centers"] = num_centers
    parameters["tree_assignment"] = tree_assignment
    parameters["query_balancing"] = query_balancing
    parameters["queries_per_tree"] = queries_per_tree

    stats["parameters"] = parameters

    t = -MPI.Wtime()

    if method == "bf": graph = points.brute_force_systolic(radius, verbosity)
    elif method == "ct": graph = points.cover_tree_systolic(radius, cover, leaf_size, verbosity)
    elif method == "gvor": graph = points.ghost_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree, verbosity)
    elif method == "cvor": graph = points.cover_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment, query_balancing, queries_per_tree, verbosity)
    elif method == "ctrma": graph = points.cover_tree_rma(radius, cover, leaf_size, verbosity)

    t += MPI.Wtime()
    maxtime = comm.reduce(t, op=MPI.MAX, root=0)

    dist_comps = comm.gather(points.dist_comps(), root=0)
    comm_times = comm.gather(points.my_comm_time(), root=0)
    comp_times = comm.gather(points.my_comp_time(), root=0)
    idle_times = comm.gather(points.my_idle_time(), root=0)

    stats["dist_comps"] = dist_comps
    stats["comm_times"] = comm_times
    stats["comp_times"] = comp_times
    stats["idle_times"] = idle_times
    stats["runtime"] = maxtime

    if method == "gvor" and query_balancing == "steal":
        stats["steal_attempts"] = comm.gather(points.steal_attempts(), root=0)
        stats["steal_successes"] = comm.gather(points.steal_successes(), root=0)
        stats["steal_services"] = comm.gather(points.steal_services(), root=0)
        stats["my_steal_comp_time"] = comm.gather(points.my_steal_comp_time(), root=0)
        stats["my_steal_time"] = comm.gather(points.my_steal_time(), root=0)
        stats["my_poll_time"] = comm.gather(points.my_poll_time(), root=0)
        stats["my_response_time"] = comm.gather(points.my_response_time(), root=0)
        stats["my_allreduce_time"] = comm.gather(points.my_allreduce_time(), root=0)

    #if myrank == 0:
    #    for i in range(nprocs):
    #        sys.stdout.write(f"[rank={i},dist_comps={format_large_number(dist_comps[i])},comp_time={comp_times[i]:.3f},comm_time={comm_times[i]:.3f},idle_time={idle_times[i]:.3f}]\n")

    sys.stdout.flush()
    comm.barrier()

    edges = graph.num_edges(num_vertices,0)
    density = edges/num_vertices

    stats["num_points"] = num_vertices
    stats["num_edges"] = edges
    stats["num_procs"] = nprocs

    if myrank == 0:
        if method == "ct":
            sys.stdout.write(f"[v0,time={maxtime:.3f},p={nprocs}] found neighbors [v={num_vertices},e={edges},e/v={density:.3f},d={format_large_number(sum(dist_comps))},c={cover:.2f},l={leaf_size},M={method}]\n")
        elif method == "ctrma":
            sys.stdout.write(f"[v0,time={maxtime:.3f},p={nprocs}] found neighbors [v={num_vertices},e={edges},e/v={density:.3f},d={format_large_number(sum(dist_comps))},c={cover:.2f},l={leaf_size},M={method}]\n")
        elif method == "bf":
            sys.stdout.write(f"[v0,time={maxtime:.3f},p={nprocs}] found neighbors [v={num_vertices},e={edges},e/v={density:.3f},d={format_large_number(sum(dist_comps))},M={method}]\n")
        elif method == "gvor":
            sys.stdout.write(f"[v0,time={maxtime:.3f},p={nprocs}] found neighbors [v={num_vertices},e={edges},e/v={density:.3f},d={format_large_number(sum(dist_comps))},c={cover:.2f},l={leaf_size},m={num_centers},M={method},A={tree_assignment},B={query_balancing},q={queries_per_tree}]\n")
        elif method == "cvor":
            sys.stdout.write(f"[v0,time={maxtime:.3f},p={nprocs}] found neighbors [v={num_vertices},e={edges},e/v={density:.3f},d={format_large_number(sum(dist_comps))},c={cover:.2f},l={leaf_size},m={num_centers},M={method},A={tree_assignment},B={query_balancing},q={queries_per_tree}]\n")

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

    if myrank == 0 and stats_file:
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)

    sys.exit(0)
