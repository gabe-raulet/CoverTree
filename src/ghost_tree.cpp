#include "ghost_tree.h"

Index GhostTree::make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made)
{
    Index edges_found = 0;

    if (finished()) return edges_found;

    if (count < 0) count = num_queries - cur_query;
    else count = std::min(count, num_queries - cur_query);

    queries_made = count;

    for (Index i = 0; i < count; ++i, ++cur_query)
    {
        Index found = tree.radius_query_indexed(points, cur_query, radius, neighs, indices);
        queries.push_back(indices[cur_query]);
        ptrs.push_back(neighs.size());
        edges_found += found;
    }

    return edges_found;
}

GhostTree::GhostTree(const CoverTree& tree, const PointVector& points, const IndexVector& indices, Index num_queries, Index id)
    : tree(tree),
      points(points),
      indices(indices),
      id(id),
      cur_query(0),
      num_queries(num_queries),
      num_points(points.num_points()),
      num_vertices(tree.num_vertices()) {}

void GhostTree::create_header_type(MPI_Datatype *MPI_GHOST_TREE_HEADER)
{
    GhostTree dummy;
    int blklens[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_INDEX, MPI_INDEX, MPI_INDEX, MPI_INDEX, MPI_INDEX};
    MPI_Aint disps[5];

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.id, &disps[0]);
    MPI_Get_address(&dummy.cur_query, &disps[1]);
    MPI_Get_address(&dummy.num_queries, &disps[2]);
    MPI_Get_address(&dummy.num_points, &disps[3]);
    MPI_Get_address(&dummy.num_vertices, &disps[4]);

    for (int i = 0; i < 5; ++i) disps[i] -= base_address;
    MPI_Type_create_struct(5, blklens, disps, types, MPI_GHOST_TREE_HEADER);
    MPI_Type_commit(MPI_GHOST_TREE_HEADER);
}

void GhostTree::allocate()
{
    /* points.res */
}
