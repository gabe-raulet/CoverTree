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
        Index found = tree.radius_query(points, cur_query, radius, neighs);
        queries.push_back(points.index(cur_query));
        ptrs.push_back(neighs.size());
        edges_found += found;
    }

    return edges_found;
}

