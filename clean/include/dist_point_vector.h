#ifndef DIST_POINT_VECTOR_H_
#define DIST_POINT_VECTOR_H_

#include "point_vector.h"
#include "dist_graph.h"
#include "cover_tree.h"
#include <mpi.h>
#include <algorithm>

#ifndef MAX_DIM
#error "MAX_DIM must be defined!"
#elif (MAX_DIM <= 0)
#error "MAX_DIM must be positive integer!"
#endif

class DistPointVector : public PointVector
{
    public:

        DistPointVector(const char *fname, MPI_Comm comm);
        ~DistPointVector();

        Index getmysize() const { return mysize; }
        Index getmyoffset() const { return myoffset; }
        Index gettotsize() const { return totsize; }

        int getmyrank() const { return myrank; }
        int getnprocs() const { return nprocs; }
        MPI_Comm getcomm() const { return comm; }

        PointVector allgather(const IndexVector& myindices, IndexVector& indices) const;
        PointVector allgather(const IndexVector& indices) const;
        PointVector gather_rma(const IndexVector& indices) const;

        Index get_rank_offset(int rank) const { return offsets[rank]; }
        Index get_rank_size(int rank) const { return (rank == nprocs-1)? totsize - offsets[nprocs-1] : offsets[rank+1]-offsets[rank]; }

        void brute_force_systolic(Real radius, DistGraph& graph, int verbosity) const;
        void cover_tree_systolic(Real radius, Real cover, Index leaf_size, DistGraph& graph, int verbosity) const;
        void cover_tree_voronoi(Real radius, Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, Index queries_per_tree, DistGraph& graph, int verbosity) const;

    protected:

        MPI_Comm comm;
        int myrank, nprocs;
        Index mysize, myoffset, totsize;
        IndexVector offsets;

        MPI_Win win;
        MPI_Datatype MPI_POINT;

        int point_owner(Index id) const { return (std::upper_bound(offsets.begin(), offsets.end(), id) - offsets.begin())-1; }

    private:

        using PointVector::dim;

        template <class Query>
        void systolic(Real radius, Query& query, DistGraph& graph, int verbosity) const;

        void build_voronoi_diagram(Index num_centers, PointVector& centers, IndexVector& centerids, IndexVector& cells, RealVector& dists, int verbosity) const;
        void find_ghost_points(Real radius, Real cover, const PointVector& centers, const IndexVector& cells, const RealVector& dists, std::vector<IndexVector>& mycellids, std::vector<IndexVector>& myghostids, int verbosity) const;

        Index compute_assignments(Index num_centers, const IndexVector& cells, const char *tree_assignment, std::vector<int>& dests, IndexVector& mycells, int verbosity) const;
        void global_point_alltoall(const std::vector<IndexVector>& ids, const std::vector<int>& dests, std::vector<PointVector>& my_cell_points, std::vector<IndexVector>& my_cell_indices, IndexVector& my_sizes, int verbosity) const;
        void build_cover_trees(std::vector<CoverTree>& mytrees, std::vector<PointVector>& my_cell_points, Real cover, Index leaf_size, int verbosity) const;
        void find_neighbors(const std::vector<CoverTree>& mytrees, const std::vector<PointVector>& my_cell_points, const std::vector<IndexVector>& my_cell_indices, const IndexVector& my_query_sizes, Real radius, Index queries_per_tree, const char *query_balancing, DistGraph& graph, int verbosity) const;
};

#endif
