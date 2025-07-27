#ifndef DIST_QUERY_H_
#define DIST_QUERY_H_

#include <mpi.h>
#include <deque>
#include "cover_tree.h"
#include "cell_vector.h"

struct GhostTree
{
    CoverTree tree;
    CellVector points;

    Index id;
    Index cur_query;
    Index num_queries;

    GhostTree(const CoverTree& tree, const CellVector& points, Index num_queries, Index id)
        : tree(tree), points(points), id(id), cur_query(0), num_queries(num_queries) {}

    bool finished() const { return cur_query >= num_queries; }
    Index make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made);
};


class DistQuery
{
    public:

        DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<CellVector>& my_cell_vectors, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, MPI_Comm comm, int verbosity);

        void static_balancing();

        Index my_edges_found() const { return num_local_edges_found; }

    private:

        Real radius;
        std::deque<GhostTree> myqueue;

        IndexVector myneighs;
        IndexVector myqueries;
        IndexVector myptrs;

        Index num_global_trees;
        Index num_local_trees_completed;
        Index num_local_queries_made;
        Index num_local_edges_found;

        MPI_Comm comm;
        int myrank, nprocs;
        int verbosity;

        bool make_tree_queries(GhostTree& tree, Index count);
        void report_finished(double mytime);
};

#endif
