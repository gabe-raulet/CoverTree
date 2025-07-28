#ifndef GHOST_TREE_H_
#define GHOST_TREE_H_

#include "cover_tree.h"
#include "cell_vector.h"

struct GhostTree
{
    CoverTree tree;
    CellVector points;

    Index id;
    Index cur_query;
    Index num_queries;

    GhostTree() = default;
    GhostTree(const CoverTree& tree, const CellVector& points, Index num_queries, Index id)
        : tree(tree),
          points(points),
          id(id),
          cur_query(0),
          num_queries(num_queries) {}

    bool finished() const { return cur_query >= num_queries; }
    Index make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made);
};

#endif
