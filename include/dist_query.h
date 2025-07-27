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
};


class DistQuery
{
    public:

        DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<CellVector>& my_cell_vectors, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, MPI_Comm comm);

    private:

        Real radius;

        MPI_Comm comm;
        int myrank, nprocs;

        std::deque<GhostTree> myqueue;
};

#endif
