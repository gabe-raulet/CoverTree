#ifndef GHOST_TREE_H_
#define GHOST_TREE_H_

#include "cover_tree.h"

struct GhostTree
{
    CoverTree tree;
    PointVector points;
    IndexVector indices;

    Index id;
    Index cur_query;
    Index num_queries;
    Index num_points;
    Index num_vertices;

    GhostTree() = default;
    GhostTree(const CoverTree& tree, const PointVector& points, const IndexVector& indices, Index num_queries, Index id);

    bool finished() const { return cur_query >= num_queries; }
    Index make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made);
};

#endif
