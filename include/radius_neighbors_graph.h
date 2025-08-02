#ifndef RADIUS_NEIGHBORS_GRAPH_H_
#define RADIUS_NEIGHBORS_GRAPH_H_

#include "utils.h"
#include "cover_tree.h"
#include "dist_voronoi.h"
#include "point_vector.h"
#include "dist_point_vector.h"
#include <mpi.h>

class RadiusNeighborsGraph
{
    public:

        RadiusNeighborsGraph(const DistPointVector& points, Real radius);

        Index brute_force_systolic(int verbosity);
        Index cover_tree_systolic(Real cover, Index leaf_size, int verbosity);
        Index cover_tree_voronoi(Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, int verbosity);

        void write_graph_file(const char *filename) const;

    private:

        Real radius;
        IndexVector myneighs, myqueries, myptrs;
        const DistPointVector points;

        template <class Query>
        Index systolic(Query& indexer);
};

#endif
