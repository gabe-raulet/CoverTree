#ifndef DIST_VORONOI_H_
#define DIST_VORONOI_H_

#include "utils.h"
#include "cover_tree.h"
#include "dist_point_vector.h"
#include <mpi.h>
#include <limits>

class DistVoronoi : public DistPointVector
{
    public:

        DistVoronoi(const DistPointVector& points);
        ~DistVoronoi();

        Index num_centers() const { return centers.num_points(); }

        void add_next_center();
        void add_next_centers(Index count);

        Index next_center_id() const { return next_center.id; }
        Real center_separation() const { return next_center.dist; } /* lower bound */

        Index compute_static_cyclic_assignments(std::vector<int>& dests, IndexVector& mycells) const;
        Index compute_multiway_number_partitioning_assignments(std::vector<int>& dests, IndexVector& mycells) const;

        Index gather_local_cell_ids(std::vector<IndexVector>& mycellids) const;
        Index gather_local_ghost_ids(Real radius, std::vector<IndexVector>& myghostids) const;

        void get_stats(Index& mincellsize, Index& maxcellsize, int root) const;

    private:

        using PointVector::dim;

        using DistPointVector::comm;
        using DistPointVector::myrank;
        using DistPointVector::nprocs;

        using DistPointVector::mysize;
        using DistPointVector::myoffset;
        using DistPointVector::totsize;

        PointVector centers; /* size: num_centers */
        IndexVector centerids; /* size: num_centers */

        IndexVector cells; /* size: mysize */
        RealVector dists; /* size: mysize */

        MPI_Datatype MPI_GLOBAL_POINT;
        MPI_Op MPI_ARGMAX;

        static void mpi_argmax(void*, void*, int*, MPI_Datatype*);
};

#endif
