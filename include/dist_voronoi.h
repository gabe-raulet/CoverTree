#ifndef DIST_VORONOI_H_
#define DIST_VORONOI_H_

#include "utils.h"
#include "cover_tree.h"
#include "point_vector.h"
#include "global_point.h"
#include <mpi.h>
#include <limits>

class DistVoronoi
{
    public:

        DistVoronoi(const PointVector& mypoints, Index global_seed, MPI_Comm comm);

        Index num_centers() const;

        void add_next_center();
        void add_next_centers(Index count);

        Index next_center_id() const;
        Real center_separation() const; /* lower bound */

        std::string repr() const;

    private:

        MPI_Comm comm;
        int myrank, nprocs;
        Index mysize, myoffset, totsize;

        GlobalPointVector mypoints; /* mysize */
        GlobalPointVector centers; /* num_centers */
        GlobalPoint next_center;

        MPI_Datatype MPI_GLOBAL_POINT;
        MPI_Op MPI_ARGMAX;

        static void mpi_argmax(void*, void*, int*, MPI_Datatype*);
};

#endif
