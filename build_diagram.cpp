#include <mpi.h>
#include <stdio.h>
#include <iostream>

#include "utils.h"
#include "point_vector.h"
#include "dist_voronoi.h"

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 4)
    {
        if (!myrank) fprintf(stderr, "Usage: %s <points> <centers> <radius>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    const char *fname = argv[1];
    Index centers = atoi(argv[2]);
    Real radius = atof(argv[3]);

    PointVector points; points.read_fvecs(fname, MPI_COMM_WORLD);

    {
        double t, maxtime;

        t = -MPI_Wtime();
        DistVoronoi diagram(points, 0, MPI_COMM_WORLD);
        diagram.add_next_centers(centers);
        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] %s\n", maxtime, diagram.repr().c_str());

        IndexVector mycellids, myghostids, mycellptrs, myghostptrs;

        t = -MPI_Wtime();
        diagram.gather_local_cell_ids(mycellids, mycellptrs);
        diagram.gather_local_ghost_ids(radius, myghostids, myghostptrs);
        t += MPI_Wtime();

        Index num_ghosts = myghostids.size();
        MPI_Allreduce(MPI_IN_PLACE, &num_ghosts, 1, MPI_INDEX, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] [num_ghosts=%lld]\n", maxtime, num_ghosts);
    }

    MPI_Finalize();
    return 0;
}
