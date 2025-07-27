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

    GlobalPointVector corel, faces, artificial40;

    corel.read_fvecs("scratch/datasets/corel.fvecs", MPI_COMM_WORLD);
    faces.read_fvecs("scratch/datasets/faces.fvecs", MPI_COMM_WORLD);
    artificial40.read_fvecs("scratch/datasets/artificial40.fvecs", MPI_COMM_WORLD);

    {
        double t1 = MPI_Wtime();
        DistVoronoi corel_diagram(corel, 0, MPI_COMM_WORLD);
        corel_diagram.add_next_centers(100);

        double t2 = MPI_Wtime();
        double mytime = t2-t1;
        double maxtime;

        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] %s\n", maxtime, corel_diagram.repr().c_str());
    }

    {
        double t1 = MPI_Wtime();
        DistVoronoi faces_diagram(faces, 0, MPI_COMM_WORLD);
        faces_diagram.add_next_centers(35);

        double t2 = MPI_Wtime();
        double mytime = t2-t1;
        double maxtime;

        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] %s\n", maxtime, faces_diagram.repr().c_str());
    }


    MPI_Finalize();
    return 0;
}
