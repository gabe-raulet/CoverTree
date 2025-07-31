#include "radius_neighbors_graph.h"

RadiusNeighborsGraph::RadiusNeighborsGraph(const char *filename, Real radius, MPI_Comm comm)
    : comm(comm),
      radius(radius)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    mypoints.read_fvecs(filename, myoffset, totsize, comm);
    mysize = mypoints.num_points();
}
