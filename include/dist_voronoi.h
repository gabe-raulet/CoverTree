#ifndef DIST_VORONOI_H_
#define DIST_VORONOI_H_

#include "utils.h"
#include "cover_tree.h"
#include "point_vector.h"
#include "global_point.h"
#include "dist_point_vector.h"
#include <mpi.h>
#include <limits>

class DistVoronoi : public DistPointVector
{
    public:

        using PointVector::dim;

        using DistPointVector::comm;
        using DistPointVector::myrank;
        using DistPointVector::nprocs;

        using DistPointVector::mysize;
        using DistPointVector::myoffset;
        using DistPointVector::totsize;

        DistVoronoi(const DistPointVector& points);
        ~DistVoronoi();

        Index num_centers() const { return centers.num_points(); }

        void add_next_center();
        void add_next_centers(Index count);

        Index next_center_id() const { return next_center.id; }
        Real center_separation() const { return next_center.dist; } /* lower bound */

        void gather_local_cell_ids(IndexVector& mycellids, IndexVector& ptrs) const;
        void gather_local_ghost_ids(Real radius, IndexVector& myghostids, IndexVector& ptrs) const;

        Index compute_static_cyclic_assignments(std::vector<int>& dests, IndexVector& mycells) const;
        Index compute_multiway_number_partitioning_assignments(std::vector<int>& dests, IndexVector& mycells) const;

        void gather_assigned_points(const std::vector<int>& dests, Real radius, std::vector<PointVector>& my_cell_points, std::vector<IndexVector>& my_cell_indices, IndexVector& my_query_sizes, int verbosity) const;

        void get_stats(Index& mincellsize, Index& maxcellsize, int root) const;

    private:

        PointVector centers; /* num_centers */
        IndexVector centerids; /* num_centers */
        GlobalPoint next_center;

        IndexVector cells; /* mysize */
        RealVector dists; /* mysize */

        MPI_Datatype MPI_GLOBAL_POINT;
        MPI_Op MPI_ARGMAX;

        static void mpi_argmax(void*, void*, int*, MPI_Datatype*);

        void load_alltoall_outbufs(const IndexVector& ids, const IndexVector& ptrs, const std::vector<int>& dests, GlobalPointVector& sendbuf, std::vector<int>& sendcounts, std::vector<int>& sdispls) const;
        void global_point_alltoall(const GlobalPointVector& sendbuf, const std::vector<int>& sendcounts, const std::vector<int>& sdispls, GlobalPointVector& recvbuf, MPI_Request *request) const;
};

#endif
