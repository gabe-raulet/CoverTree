#include <mpi.h>
#include <stdio.h>

#include "utils.h"
#include "point_vector.h"
#include "global_point.h"
#include "global_point_vector.h"
#include "dist_voronoi.h"
#include "cover_tree.h"

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* PointVector points(100); */
    /* CoverTree tree(PointVector(100)); */
    /* GlobalPoint p; */

    /* DistVoronoi diagram(PointVector(100), 0, MPI_COMM_WORLD); */
    /* GlobalPointVector points(3); */

    MPI_Finalize();
    return 0;
}
