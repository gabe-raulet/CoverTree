#ifndef RADIUS_NEIGHBORS_GRAPH_H_
#define RADIUS_NEIGHBORS_GRAPH_H_

#include "utils.h"
#include "point_vector.h"
#include <mpi.h>

class RadiusNeighborsGraph
{
    public:

        RadiusNeighborsGraph(const char *filename, Real radius, MPI_Comm comm);



    private:

        MPI_Comm comm;
        int myrank, nprocs;

        Real radius;
        PointVector mypoints;
        Index mysize, myoffset, totsize;
        IndexVector myneighs, myqueries, myptrs;


};

#endif
