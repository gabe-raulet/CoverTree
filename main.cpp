#include <mpi.h>
#include <stdio.h>
#include <iostream>

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

    PointVector corel, faces, artificial40;

    corel.read_fvecs("scratch/datasets/corel.fvecs");
    faces.read_fvecs("scratch/datasets/faces.fvecs");
    artificial40.read_fvecs("scratch/datasets/artificial40.fvecs");

    printf("corel: %s\n", corel.repr().c_str());
    printf("faces: %s\n", faces.repr().c_str());
    printf("artificial40: %s\n", artificial40.repr().c_str());

    MPI_Finalize();
    return 0;
}
