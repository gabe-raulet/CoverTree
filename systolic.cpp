#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>

#include "utils.h"
#include "point_vector.h"
#include "cover_tree.h"

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Real radius;
    const char *infile;

    if (argc != 3)
    {
        if (!myrank) fprintf(stderr, "Usage: %s <points> <radius>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    infile = argv[1];
    radius = atof(argv[2]);

    double mytime, maxtime;

    /*
     * Read input points file
     */

    mytime = -MPI_Wtime();
    PointVector mypoints; mypoints.read_fvecs(infile, MPI_COMM_WORLD);
    mytime += MPI_Wtime();

    Index totsize;
    Index mysize = mypoints.num_points();
    int dim = mypoints.num_dimensions();

    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!myrank) printf("[v1,time=%.3f] read file '%s' [size=%lld,dim=%d]\n", maxtime, infile, totsize, dim);
    fflush(stdout);

    IndexVector myneighs, myqueries, myptrs;

    mytime = -MPI_Wtime();
    CoverTree tree;
    tree.build(mypoints, 1.3, 20);
    tree.distributed_query(radius, mypoints, myneighs, myqueries, myptrs, MPI_COMM_WORLD, 1);
    mytime += MPI_Wtime();

    Index totedges;
    Index myedges = myneighs.size();
    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myedges, &totedges, 1, MPI_INDEX, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!myrank) printf("[v1,time=%.3f] completed queries [edges=%lld,density=%.3f]\n", maxtime, totedges, (totedges+0.0)/totsize);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
