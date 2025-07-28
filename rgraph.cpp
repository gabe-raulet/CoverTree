#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>

#include "utils.h"
#include "point_vector.h"
#include "cell_vector.h"
#include "dist_voronoi.h"
#include "dist_query.h"

struct Parameters
{
    const char *infile;
    const char *outfile;
    Index leaf_size, num_centers, queries_per_tree;
    std::string assignment_method;
    Real cover, radius;
    int verbosity;
    int pinned;

    Parameters();

    void parse_cmdline(int argc, char *argv[], MPI_Comm comm);
};

int main_mpi(const Parameters& parameters, MPI_Comm comm);
int main(int argc, char *argv[])
{
    Parameters parameters;

    MPI_Init(&argc, &argv);
    parameters.parse_cmdline(argc, argv, MPI_COMM_WORLD);
    int err = main_mpi(parameters, MPI_COMM_WORLD);
    MPI_Finalize();
    return err;
}

int main_mpi(const Parameters& parameters, MPI_Comm comm)
{
    double mytime, maxtime, t;

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    const char *infile = parameters.infile;
    const char *outfile = parameters.outfile;
    Index num_centers = parameters.num_centers;
    Index queries_per_tree = parameters.queries_per_tree;
    Real radius = parameters.radius;
    Real cover = parameters.cover;
    Index leaf_size = parameters.leaf_size;
    std::string assignment_method = parameters.assignment_method;
    int verbosity = parameters.verbosity;

    mytime = -MPI_Wtime();
    PointVector mypoints; mypoints.read_fvecs(infile, comm);
    mytime += MPI_Wtime();

    double tottime = -MPI_Wtime();

    Index totsize;
    Index mysize = mypoints.num_points();
    int dim = mypoints.num_dimensions();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] read file '%s' [size=%lld,dim=%d]\n", maxtime, infile, totsize, dim); fflush(stdout); }
    }

    mytime = -MPI_Wtime();
    DistVoronoi diagram(mypoints, 0, comm);
    diagram.add_next_centers(num_centers);
    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        Index mincellsize, maxcellsize;
        diagram.get_stats(mincellsize, maxcellsize, 0);

        if (!myrank) { printf("[v1,time=%.3f] found %lld centers [separation=%.3f,minsize=%lld,maxsize=%lld,avgsize=%.3f]\n", maxtime, num_centers, diagram.center_separation(), mincellsize, maxcellsize, (totsize+0.0)/num_centers); fflush(stdout); }
    }

    IndexVector mycellids, myghostids, mycellptrs, myghostptrs;

    mytime = -MPI_Wtime();
    diagram.gather_local_cell_ids(mycellids, mycellptrs);
    diagram.gather_local_ghost_ids(radius, myghostids, myghostptrs);
    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        Index num_ghosts = myghostids.size();

        const void *sendbuf = myrank == 0? MPI_IN_PLACE : &num_ghosts;
        MPI_Reduce(sendbuf, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) { printf("[v1,time=%.3f] found %lld ghost points [avgprocsize=%.3f]\n", maxtime, num_ghosts, (num_ghosts+0.0)/nprocs); fflush(stdout); }
    }

    Index s;
    IndexVector mycells;
    std::vector<int> dests; /* tree-to-rank assignments */

    mytime = -MPI_Wtime();
    if      (assignment_method == "cyclic") s = diagram.compute_static_cyclic_assignments(dests, mycells);
    else if (assignment_method == "multiway") s = diagram.compute_multiway_number_partitioning_assignments(dests, mycells);
    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] computed tree-to-rank assignments\n", maxtime); fflush(stdout); }
    }

    std::vector<int> cell_sendcounts, cell_recvcounts, cell_sdispls, cell_rdispls;
    std::vector<int> ghost_sendcounts, ghost_recvcounts, ghost_sdispls, ghost_rdispls;

    std::vector<GlobalPoint> cell_sendbuf, cell_recvbuf;
    std::vector<GlobalPoint> ghost_sendbuf, ghost_recvbuf;

    mytime = -MPI_Wtime();
    diagram.load_alltoall_outbufs(mycellids, mycellptrs, dests, cell_sendbuf, cell_sendcounts, cell_sdispls);
    diagram.load_alltoall_outbufs(myghostids, myghostptrs, dests, ghost_sendbuf, ghost_sendcounts, ghost_sdispls);
    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] loaded alltoall outbufs\n", maxtime); fflush(stdout); }
    }

    mytime = -MPI_Wtime();

    MPI_Request reqs[2];

    MPI_Datatype MPI_GLOBAL_POINT;
    GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, dim);

    global_point_alltoall(cell_sendbuf, cell_sendcounts, cell_sdispls, MPI_GLOBAL_POINT, cell_recvbuf, comm, &reqs[0]);
    global_point_alltoall(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, MPI_GLOBAL_POINT, ghost_recvbuf, comm, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    MPI_Type_free(&MPI_GLOBAL_POINT);

    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] alltoall exchange\n", maxtime); fflush(stdout); }
    }

    mytime = -MPI_Wtime();

    IndexVector my_query_sizes(s,0);
    std::vector<CellVector> my_cell_vectors(s, CellVector(dim));

    build_local_cell_vectors(cell_recvbuf, ghost_recvbuf, my_cell_vectors, my_query_sizes);

    mytime += MPI_Wtime();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] built local cell vectors\n", maxtime); fflush(stdout); }
    }

    mytime = -MPI_Wtime();

    std::vector<CoverTree> mytrees(s);

    for (Index i = 0; i < s; ++i)
    {
        t = -MPI_Wtime();
        mytrees[i].build(my_cell_vectors[i], cover, leaf_size);
        t += MPI_Wtime();

        if (verbosity > 2) { printf("[v3,rank=%d,time=%.3f] built cover tree [locid=%lld,globid=%lld,vertices=%lld]\n", myrank, t, i, mycells[i], mytrees[i].num_vertices()); fflush(stdout); }
    }

    mytime += MPI_Wtime();

    if (verbosity > 1)
    {
        printf("[v2,rank=%d,time=%.3f] completed %lld local trees\n", myrank, mytime, s);
        fflush(stdout);
    }

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] built %lld cover trees\n", maxtime, num_centers); fflush(stdout); }
    }


    mytime = -MPI_Wtime();
    DistQuery dist_query(mytrees, my_cell_vectors, my_query_sizes, mycells, radius, comm, verbosity);
    dist_query.static_balancing();
    mytime += MPI_Wtime();

    tottime += MPI_Wtime();

    if (verbosity > 0)
    {
        Index edges;
        Index myedges = dist_query.my_edges_found();

        MPI_Reduce(&myedges, &edges, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) { printf("[v1,time=%.3f] completed queries [edges=%lld,density=%.3f]\n", mytime, edges, (edges+0.0)/totsize); fflush(stdout); }
    }

    if (outfile)
    {
        mytime = -MPI_Wtime();
        dist_query.write_to_file(outfile);
        mytime += MPI_Wtime();

        if (verbosity > 0)
        {
            MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!myrank) { printf("[v1,time=%.3f] wrote graph to file '%s'\n", maxtime, outfile); fflush(stdout); }
        }
    }

    MPI_Reduce(myrank == 0? MPI_IN_PLACE : &tottime, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!myrank) printf("\n[total_runtime=%.3f,nprocs=%d]\n\n", tottime, nprocs);

    return 0;
}

Parameters::Parameters()
    : infile(NULL),
      outfile(NULL),
      leaf_size(10),
      num_centers(50),
      queries_per_tree(-1),
      assignment_method("cyclic"),
      cover(1.3),
      radius(-1.),
      verbosity(1),
      pinned(0) {}

void Parameters::parse_cmdline(int argc, char *argv[], MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <points> -r <radius>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT cover tree base [%.2f]\n", cover);
            fprintf(stderr, "         -l INT   leaf size [%lld]\n", leaf_size);
            fprintf(stderr, "         -m INT   centers per processor [%lld]\n", num_centers);
            fprintf(stderr, "         -M INT   pinned centers (overrides -m)\n");
            fprintf(stderr, "         -q INT   queries per tree [%lld]\n", queries_per_tree);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -o FILE  output sparse graph\n");
            fprintf(stderr, "         -a STR   cell assignment method (one of: cyclic, multiway) [%s]\n", assignment_method.c_str());
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Abort(comm, err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:m:M:v:o:i:r:q:a:h")) >= 0)
    {

        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'm') num_centers = nprocs * atoi(optarg);
        else if (c == 'M') { num_centers = atoi(optarg); pinned = 1; }
        else if (c == 'q') queries_per_tree = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'a') assignment_method = std::string(optarg);
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!infile || radius < 0) usage(1, myrank == 0);
}
