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

struct BruteForceQuery
{
    Index operator()(const PointVector& mypoints, const AtomVector& curpoints, int dim, Index cursize, Index curoffset, Index mysize, Index myoffset, Real radius, IndexVector& myqueries, IndexVector& myneighs, IndexVector& myptrs)
    {
        Index num_edges = 0;

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

        return num_edges;
    }
};

struct CoverTreeQuery
{
    CoverTree mytree;

    CoverTreeQuery(const PointVector& mypoints, Real cover, Index leaf_size)
    {
        mytree.build(mypoints, cover, leaf_size);
    }

    Index operator()(const PointVector& mypoints, const AtomVector& curpoints, int dim, Index cursize, Index curoffset, Index mysize, Index myoffset, Real radius, IndexVector& myqueries, IndexVector& myneighs, IndexVector& myptrs)
    {
        Index num_edges = 0;

        for (Index j = 0; j < cursize; ++j)
        {
            myqueries.push_back(j+curoffset);

            IndexVector neighs;
            num_edges += mytree.radius_query(mypoints, &curpoints[j*dim], radius, neighs);
            for (Index id : neighs) myneighs.push_back(id+myoffset);
            myptrs.push_back(myneighs.size());
        }

        return num_edges;
    }
};

Index RadiusNeighborsGraph::brute_force_systolic()
{
    BruteForceQuery indexer;
    return systolic(indexer);
}

Index RadiusNeighborsGraph::cover_tree_systolic(Real cover, Index leaf_size)
{
    CoverTreeQuery indexer(mypoints, cover, leaf_size);
    return systolic(indexer);
}

template <class Query>
Index RadiusNeighborsGraph::systolic(Query& indexer)
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

        num_edges += indexer(mypoints, curpoints, dim, cursize, curoffset, mysize, myoffset, radius, myqueries, myneighs, myptrs);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        cur = (cur+1)%nprocs;
        std::swap(curpoints, nextpoints);
    }

    MPI_Allreduce(MPI_IN_PLACE, &num_edges, 1, MPI_INDEX, MPI_SUM, comm);

    return num_edges;
}

void RadiusNeighborsGraph::write_graph_file(const char *filename) const
{
    std::ostringstream ss, ss2;
    Index num_edges, my_num_edges = 0;
    Index my_num_queries = myqueries.size();

    for (Index i = 0; i < my_num_queries; ++i)
        for (Index p = myptrs[i]; p < myptrs[i+1]; ++p)
            if (myqueries[i] != myneighs[p])
            {
                ss2 << (myqueries[i]+1) << " " << (myneighs[p]+1) << "\n";
                my_num_edges++;
            }

    for (Index i = 0; i < mysize; ++i)
    {
        ss2 << (i+myoffset+1) << " " << (i+myoffset+1) << "\n";
        my_num_edges++;
    }

    MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);

    if (!myrank) ss << "% " << totsize << " " << totsize << " " << num_edges << "\n" << ss2.str();
    else std::swap(ss, ss2);

    auto sbuf = ss.str();
    std::vector<char> buf(sbuf.begin(), sbuf.end());

    MPI_Offset mycount = buf.size(), fileoffset, filesize;
    MPI_Exscan(&mycount, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    int truncate = 0;

    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &filesize);
    truncate = (filesize > 0);
    MPI_Bcast(&truncate, 1, MPI_INT, 0, comm);
    if (truncate) MPI_File_set_size(fh, 0);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
