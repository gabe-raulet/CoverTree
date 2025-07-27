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
        ~DistVoronoi();

        Index num_centers() const { return centers.num_points(); }

        void add_next_center();
        void add_next_centers(Index count);

        Index next_center_id() const { return next_center.id; }
        Real center_separation() const { return next_center.dist; } /* lower bound */

        std::string repr() const;

        void gather_local_cell_ids(IndexVector& mycellids, IndexVector& ptrs) const;
        void gather_local_ghost_ids(Real radius, IndexVector& myghostids, IndexVector& ptrs) const;

        void load_alltoall_outbufs(const IndexVector& ids, const IndexVector& ptrs, const std::vector<int>& dests, std::vector<GlobalPoint>& sendbuf, std::vector<int>& sendcounts, std::vector<int>& sdispls) const;

    private:

        PointVector centers; /* num_centers */
        IndexVector centerids; /* num_centers */
        GlobalPoint next_center;

        PointVector mypoints; /* mysize */
        IndexVector cells; /* mysize */
        RealVector dists; /* mysize */

        MPI_Comm comm;
        int myrank, nprocs;
        Index mysize, myoffset, totsize;

        MPI_Datatype MPI_GLOBAL_POINT;
        MPI_Op MPI_ARGMAX;

        static void mpi_argmax(void*, void*, int*, MPI_Datatype*);
};

#endif
