#ifndef DIST_GRAPH_H_
#define DIST_GRAPH_H_

#include "utils.h"
#include <mpi.h>

/*
 * This class assumes self-loops always exist.
 */

class DistGraph
{
    public:

        DistGraph(MPI_Comm comm);

        void add_neighbors(Index query, const IndexVector& neighbors, Index offset = 0);
        void write_edge_file(Index num_vertices, const char *filename) const;

    private:

        IndexVector myqueries;
        IndexVector myneighs;
        IndexVector myptrs;

        MPI_Comm comm;
        int myrank, nprocs;
};

#endif
