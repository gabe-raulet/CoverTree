#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <string.h>
#include <unistd.h>

#include "utils.h"
#include "timer.h"
#include "point_vector.h"
#include "dist_point_vector.h"
#include "dist_graph.h"

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
const char *tree_assignment = "multiway";
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
    Timer section_timer(comm, 0), main_timer(comm, 0);

    section_timer.start();
    DistPointVector points(infile, comm);
    section_timer.stop();

    Index num_vertices = points.gettotsize();

    if (verbosity >= 1 && !myrank)
    {
        printf("[v1,%s] Read file '%s' [size=%lld,dim=%d]\n", section_timer.repr().c_str(), infile, num_vertices, points.num_dimensions());
    }

    DistGraph graph(comm);
    points.brute_force_systolic(radius, graph, 1);
    /* points.cover_tree_systolic(radius, cover, leaf_size, graph, 1); */

    graph.write_edge_file(num_vertices, "edges.mtx");

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

    if (strcmp(query_balancing, "static") && strcmp(query_balancing, "steal"))
    {
        if (!myrank) fprintf(stderr, "error: '%s' is an invalid query balancing method!\n", query_balancing);
        MPI_Finalize();
        std::exit(1);
    }
}
