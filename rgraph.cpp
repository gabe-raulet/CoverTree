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

struct Parameters
{
    const char *infile;
    const char *outfile;
    Index leaf_size, num_centers;
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
    double mytime, maxtime;

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    const char *infile = parameters.infile;
    Index num_centers = parameters.num_centers;
    Real radius = parameters.radius;
    int verbosity = parameters.verbosity;

    mytime = -MPI_Wtime();
    PointVector mypoints; mypoints.read_fvecs(infile, comm);
    mytime += MPI_Wtime();

    Index totsize;
    Index mysize = mypoints.num_points();

    if (verbosity > 0)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) printf("[time=%.3f] read file '%s' [size=%lld,dim=%d]\n", maxtime, infile, totsize, mypoints.num_dimensions());
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

        if (!myrank) printf("[time=%.3f] found %lld centers [separation=%.3f,minsize=%lld,maxsize=%lld,avgsize=%.3f]\n", maxtime, num_centers, diagram.center_separation(), mincellsize, maxcellsize, (totsize+0.0)/num_centers);
    }

    return 0;
}

Parameters::Parameters()
    : infile(NULL),
      outfile(NULL),
      leaf_size(10),
      num_centers(50),
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
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -o FILE  output sparse graph\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Abort(comm, err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:m:M:v:o:i:r:h")) >= 0)
    {
        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'm') num_centers = nprocs * atoi(optarg);
        else if (c == 'M') { num_centers = atoi(optarg); pinned = 1; }
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!infile || radius < 0) usage(1, myrank == 0);
}
