#include <mpi.h>
#include <stdio.h>
#include <iostream>

#include "utils.h"
#include "point_vector.h"
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

    CoverTree corel_tree(corel);
    CoverTree faces_tree(faces);
    CoverTree artificial40_tree(artificial40);

    for (Index leaf_size : {1,2,4,8,16,32,64,128,256,512,1024,2048,4096})
    {
        corel_tree.build(1.3, leaf_size);
        printf("corel_tree(leaf_size=%lld) = %s\n", leaf_size, corel_tree.repr().c_str());
    }

    /* corel_tree.build(1.3, 10); */
    /* faces_tree.build(1.3, 10); */
    /* artificial40_tree.build(1.3, 10); */

    /* printf("corel_tree: %s\n", corel_tree.repr().c_str()); */
    /* printf("faces_tree: %s\n", faces_tree.repr().c_str()); */
    /* printf("artificial40_tree: %s\n", artificial40_tree.repr().c_str()); */

    MPI_Finalize();
    return 0;
}
