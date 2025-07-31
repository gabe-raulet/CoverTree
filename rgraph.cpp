#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>

#include "utils.h"
#include "point_vector.h"
#include "dist_voronoi.h"
#include "dist_query.h"
#include "radius_neighbors_graph.h"

MPI_Comm comm;
int myrank, nprocs;

Real radius = -1;
const char *infile = NULL;

Real cover = 1.3;
Index leaf_size = 10;
Index queries_per_tree = -1;
Index num_centers = 25;
int verbosity = 1;

const char *outfile = NULL;
const char *tree_assignment = "static";
const char *query_balancing = "static";

void parse_cmdline(int argc, char *argv[]);

int main_mpi(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    parse_cmdline(argc, argv);
    int err = main_mpi(argc, argv);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return err;
}

int main_mpi(int argc, char *argv[])
{
    double mytime = 0, maxtime, t;

    RadiusNeighborsGraph rnng(infile, radius, comm);

    t = -MPI_Wtime();
    Index num_edges = rnng.brute_force_systolic();
    t += MPI_Wtime();
    mytime += t;

    Index num_points = rnng.gettotsize();
    Real density = (num_edges+0.0) / num_points;

    if (!myrank) printf("[time=%.3f] [num_points=%lld,num_edges=%lld,density=%.3f]\n", mytime, num_points, num_edges, density);

    if (outfile) rnng.write_graph_file(outfile);

    //int dim;
    //Index mysize, totsize;
    //PointVector mypoints;

    //double tottime, mytime, maxtime, t;

    ///*
    // * Read input points file
    // */

    //mytime = -MPI_Wtime();
    //mypoints.read_fvecs(infile, comm);
    //mytime += MPI_Wtime();

    //mysize = mypoints.num_points();
    //dim = mypoints.num_dimensions();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    MPI_Reduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] read file '%s' [size=%lld,dim=%d]\n", maxtime, infile, totsize, dim);
    //    fflush(stdout);
    //}

    //MPI_Barrier(comm);
    //tottime = -MPI_Wtime();

    ///*
    // * Partition points into Voronoi cells
    // */

    //mytime = -MPI_Wtime();
    //DistVoronoi diagram(mypoints, 0, comm);
    //diagram.add_next_centers(num_centers);
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    //    Index mincellsize, maxcellsize;
    //    diagram.get_stats(mincellsize, maxcellsize, 0);

    //    if (!myrank) printf("[v1,time=%.3f] found %lld centers [separation=%.3f,minsize=%lld,maxsize=%lld,avgsize=%.3f]\n", maxtime, num_centers, diagram.center_separation(), mincellsize, maxcellsize, (totsize+0.0)/num_centers);
    //    fflush(stdout);
    //}

    ///*
    // * Gather cell points and find ghost points
    // */

    //IndexVector mycellids, myghostids, mycellptrs, myghostptrs;

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //diagram.gather_local_cell_ids(mycellids, mycellptrs);
    //diagram.gather_local_ghost_ids(radius, myghostids, myghostptrs);
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    Index num_ghosts = myghostids.size();

    //    const void *sendbuf = myrank == 0? MPI_IN_PLACE : &num_ghosts;
    //    MPI_Reduce(sendbuf, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    //    if (!myrank) printf("[v1,time=%.3f] found %lld ghost points [avgprocsize=%.3f]\n", maxtime, num_ghosts, (num_ghosts+0.0)/nprocs);
    //    fflush(stdout);
    //}

    ///*
    // * Compute tree-to-rank assignments
    // */

    //Index s;
    //IndexVector mycells;
    //std::vector<int> dests; /* tree-to-rank assignments */

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //if      (!strcmp(tree_assignment, "static")) s = diagram.compute_static_cyclic_assignments(dests, mycells);
    //else if (!strcmp(tree_assignment, "multiway")) s = diagram.compute_multiway_number_partitioning_assignments(dests, mycells);
    //else throw std::runtime_error("invalid assignments_methods selected!");
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] computed tree-to-rank assignments\n", maxtime);
    //    fflush(stdout);
    //}

    ///*
    // * Load alltoall buffers
    // */

    //std::vector<int> cell_sendcounts, cell_recvcounts, cell_sdispls, cell_rdispls;
    //std::vector<int> ghost_sendcounts, ghost_recvcounts, ghost_sdispls, ghost_rdispls;

    //GlobalPointVector cell_sendbuf, cell_recvbuf;
    //GlobalPointVector ghost_sendbuf, ghost_recvbuf;

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //diagram.load_alltoall_outbufs(mycellids, mycellptrs, dests, cell_sendbuf, cell_sendcounts, cell_sdispls);
    //diagram.load_alltoall_outbufs(myghostids, myghostptrs, dests, ghost_sendbuf, ghost_sendcounts, ghost_sdispls);
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] loaded alltoall outbufs\n", maxtime);
    //    fflush(stdout);
    //}

    ///*
    // * Exchange points alltoall
    // */

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();

    //MPI_Request reqs[2];

    //MPI_Datatype MPI_GLOBAL_POINT;
    //GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, dim);

    //global_point_alltoall(cell_sendbuf, cell_sendcounts, cell_sdispls, MPI_GLOBAL_POINT, cell_recvbuf, comm, &reqs[0]);
    //global_point_alltoall(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, MPI_GLOBAL_POINT, ghost_recvbuf, comm, &reqs[1]);

    //MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    //MPI_Type_free(&MPI_GLOBAL_POINT);

    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] alltoall exchange\n", maxtime);
    //    fflush(stdout);
    //}

    ///*
    // * Build local cell vectors
    // */

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();

    //IndexVector my_query_sizes(s,0);
    //std::vector<PointVector> my_cell_vectors(s, PointVector(dim));
    //std::vector<IndexVector> my_cell_indices(s);

    //build_local_cell_vectors(cell_recvbuf, ghost_recvbuf, my_cell_vectors, my_cell_indices, my_query_sizes, false);

    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) { printf("[v1,time=%.3f] built local cell vectors\n", maxtime); fflush(stdout); }
    //}

    ///*
    // * Build local cover trees
    // */

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();

    //std::vector<CoverTree> mytrees(s);

    //for (Index i = 0; i < s; ++i)
    //{
    //    t = -MPI_Wtime();
    //    mytrees[i].build(my_cell_vectors[i], cover, leaf_size);
    //    t += MPI_Wtime();

    //    if (verbosity >= 3) printf("[v3,rank=%d,time=%.3f] built cover tree [id=%lld,points=%lld,vertices=%lld]\n", myrank, t, mycells[i], my_cell_vectors[i].num_points(), mytrees[i].num_vertices());

    //    fflush(stdout);
    //}

    //mytime += MPI_Wtime();

    //if (verbosity >= 2)
    //{
    //    printf("[v2,rank=%d,time=%.3f] completed %lld local trees\n", myrank, mytime, s);
    //    fflush(stdout);
    //}

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] built %lld cover trees\n", maxtime, num_centers);
    //    fflush(stdout);
    //}

    ///*
    // * Compute epsilon neighbors
    // */

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //DistQuery dist_query(mytrees, my_cell_vectors, my_cell_indices, my_query_sizes, mycells, radius, dim, comm, verbosity);

    //if      (!strcmp(query_balancing, "static") || nprocs == 1) dist_query.static_balancing();
    //else if (!strcmp(query_balancing, "steal")) dist_query.random_stealing(queries_per_tree);
    //else throw std::runtime_error("Invalid balancing_method selected!");

    //mytime += MPI_Wtime();
    //tottime += MPI_Wtime();

    //Index edges;
    //Index myedges = dist_query.my_edges_found();
    //MPI_Reduce(&myedges, &edges, 1, MPI_INDEX, MPI_SUM, 0, comm);

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[v1,time=%.3f] completed queries [edges=%lld,density=%.3f]\n", mytime, edges, (edges+0.0)/totsize);
    //    fflush(stdout);
    //}

    //if (outfile)
    //{
    //    MPI_Barrier(comm);
    //    mytime = -MPI_Wtime();
    //    dist_query.write_to_file(outfile);
    //    mytime += MPI_Wtime();

    //    if (verbosity > 0)
    //    {
    //        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //        if (!myrank) printf("[v1,time=%.3f] wrote graph to file '%s'\n", maxtime, outfile);
    //        fflush(stdout);
    //    }
    //}

    //MPI_Reduce(myrank == 0? MPI_IN_PLACE : &tottime, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    //if (verbosity >= 1)
    //{
    //    if (!myrank) printf("\n[total_runtime=%.3f,nprocs=%d]\n\n", tottime, nprocs);
    //}
    //else
    //{
    //    if (!myrank)
    //    {
    //        printf("[time=%.3f,nprocs=%d,edges=%lld,radius=%.3f,cover=%.3f,leaf_size=%lld,centers=%lld,queries_per_tree=%lld,assignment=%s,balancing=%s]\n",
    //                 tottime, nprocs, edges, radius, cover, leaf_size, num_centers, queries_per_tree, tree_assignment, query_balancing);
    //    }
    //}

    //fflush(stdout);
    return 0;
}

void parse_cmdline(int argc, char *argv[])
{
    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <points> -r <radius>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT cover tree base [%.2f]\n", cover);
            fprintf(stderr, "         -l INT   leaf size [%lld]\n", leaf_size);
            fprintf(stderr, "         -m INT   num centers [%lld]\n", num_centers);
            fprintf(stderr, "         -q INT   queries per tree [%lld]\n", queries_per_tree);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -o FILE  output sparse graph\n");
            fprintf(stderr, "         -A STR   tree assignment method (one of: static, multiway) [%s]\n", tree_assignment);
            fprintf(stderr, "         -B STR   load balancing method (one of: static, steal) [%s]\n", query_balancing);
            fprintf(stderr, "         -F       fix total centers\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    bool fix_num_centers = false;

    int c;
    while ((c = getopt(argc, argv, "c:l:m:Fv:o:i:r:q:A:B:h")) >= 0)
    {

        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'm') num_centers = atoi(optarg);
        else if (c == 'q') queries_per_tree = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'A') tree_assignment = optarg;
        else if (c == 'B') query_balancing = optarg;
        else if (c == 'F') fix_num_centers = true;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!fix_num_centers) num_centers *= nprocs;

    if (!infile || radius < 0) usage(1, myrank == 0);

    if (strcmp(tree_assignment, "static") && strcmp(tree_assignment, "multiway"))
    {
        if (!myrank) fprintf(stderr, "error: '%s' is an invalid tree assignment method!\n", tree_assignment);
        MPI_Finalize();
        std::exit(1);
    }
    else if (strcmp(query_balancing, "static") && strcmp(query_balancing, "steal"))
    {
        if (!myrank) fprintf(stderr, "error: '%s' is an invalid query balancing method!\n", query_balancing);
        MPI_Finalize();
        std::exit(1);
    }
}
