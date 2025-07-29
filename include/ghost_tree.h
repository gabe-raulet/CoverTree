#ifndef GHOST_TREE_H_
#define GHOST_TREE_H_

#include "cover_tree.h"

struct GhostTreeHeader
{
    Index id;
    Index cur_query;
    Index num_queries;
    Index num_points;
    Index num_vertices;

    static void create_header_type(MPI_Datatype *MPI_GHOST_TREE_HEADER);
};

struct GhostTree
{
    CoverTree tree;
    PointVector points;
    IndexVector indices;
    GhostTreeHeader header;

    GhostTree() = default;
    GhostTree(const CoverTree& tree, const PointVector& points, const IndexVector& indices, Index num_queries, Index id);

    bool finished() const { return header.cur_query >= header.num_queries; }
    Index make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made);

    void allocate(const GhostTreeHeader& recv_header, int dim);
};

#endif
