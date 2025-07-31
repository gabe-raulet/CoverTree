#include "radius_neighbors_graph.h"
#include <numeric>

RadiusNeighborsGraph::RadiusNeighborsGraph(const char *filename, Real radius, MPI_Comm comm)
    : comm(comm),
      radius(radius),
      myptrs({0})
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    mypoints.read_fvecs(filename, myoffset, totsize, comm);
    mysize = mypoints.num_points();
}

Index RadiusNeighborsGraph::brute_force_systolic()
{
    IndexVector allsizes(nprocs), alloffsets(nprocs);
    allsizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, allsizes.data(), 1, MPI_INDEX, comm);

    std::exclusive_scan(allsizes.begin(), allsizes.end(), alloffsets.begin(), (Index)0);

    MPI_Request reqs[2];

    int next = (myrank+1)%nprocs;
    int prev = (myrank-1+nprocs)%nprocs;
    int cur = myrank;

    AtomVector curpoints = mypoints.copy_atoms();
    AtomVector nextpoints;

    int dim = mypoints.num_dimensions();

    Index num_edges = 0;

    for (int step = 0; step < nprocs; ++step)
    {
        Index numrecv = allsizes[(cur+1)%nprocs];
        Index numsend = allsizes[cur];

        int recvcount = numrecv*dim;
        int sendcount = numsend*dim;

        nextpoints.resize(recvcount);

        MPI_Irecv(nextpoints.data(), recvcount, MPI_ATOM, next, myrank, comm, &reqs[0]);
        MPI_Isend(curpoints.data(), sendcount, MPI_ATOM, prev, prev, comm, &reqs[1]);

        Index cursize = allsizes[cur];
        Index curoffset = alloffsets[cur];

        for (Index j = 0; j < cursize; ++j)
        {
            myqueries.push_back(j+curoffset);

            for (Index i = 0; i < mysize; ++i)
                if (mypoints.distance(i, &curpoints[j*dim]) <= radius)
                {
                    myneighs.push_back(i+myoffset);
                    num_edges++;
                }

            myptrs.push_back(myneighs.size());
        }

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        cur = (cur+1)%nprocs;
        std::swap(curpoints, nextpoints);
    }

    MPI_Allreduce(MPI_IN_PLACE, &num_edges, 1, MPI_INDEX, MPI_SUM, comm);

    return num_edges;
}
