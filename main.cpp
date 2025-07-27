#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>

#include "utils.h"
#include "point_vector.h"
#include "global_point.h"
#include "dist_voronoi.h"
#include "cover_tree.h"

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (!myrank)
    {
        PointVector points;
        points.read_fvecs("scratch/datasets/faces.fvecs");

        std::default_random_engine gen(10);
        IndexVector ids(points.num_points());
        std::iota(ids.begin(), ids.end(), (Index)0);
        std::shuffle(ids.begin(), ids.end(), gen);

        ids.resize(100);

        PointVector subset = points.gather(ids);

        subset.write_fvecs("subset.fvecs");
    }

    MPI_Finalize();
    return 0;
}
