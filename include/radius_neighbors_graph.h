#ifndef RADIUS_NEIGHBORS_GRAPH_H_
#define RADIUS_NEIGHBORS_GRAPH_H_

#include "utils.h"
#include "point_vector.h"
#include "cover_tree.h"
#include <mpi.h>

class RadiusNeighborsGraph
{
    public:

        RadiusNeighborsGraph(const char *filename, Real radius, MPI_Comm comm);

        Index brute_force_systolic(int verbosity);
        Index cover_tree_systolic(Real cover, Index leaf_size, int verbosity);
        Index cover_tree_voronoi(Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, int verbosity);

        Index getmysize() const { return mysize; }
        Index getmyoffset() const { return myoffset; }
        Index gettotsize() const { return totsize; }

        void write_graph_file(const char *filename) const;

    private:

        MPI_Comm comm;
        int myrank, nprocs;

        Real radius;
        PointVector mypoints;
        Index mysize, myoffset, totsize;
        IndexVector myneighs, myqueries, myptrs;

        template <class Query>
        Index systolic(Query& indexer);
};

#endif
