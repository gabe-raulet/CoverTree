#include "dist_point_vector.h"
#include "cover_tree.h"
#include <assert.h>
#include <filesystem>
#include <limits>

void DistPointVector::init_comm()
{
    /*
     * Assumes `comm` has been initialized!
     */

    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
}

void DistPointVector::init_offsets()
{
    /*
     * Assumes `init_comm` has been called, and that `mysize` and
     * `PointVector::dim` have been initialized!
     */

    IndexVector sizes(nprocs);
    sizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, sizes.data(), 1, MPI_INDEX, comm);

    offsets.resize(nprocs);
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), (Index)0);

    totsize = offsets.back() + sizes.back();
    myoffset = offsets[myrank];

    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);
}

void DistPointVector::init_window()
{
    /*
     * Assumes `init_comm` and `init_offsets` have been called, and that PointVector::atoms
     * has been allocated and is consistent with all instance variables (besides `win` and `MPI_POINT`).
     * Essentially, only call this after every other variable has been initialized in both PointVector
     * and DistPointVector.
     */

    int dim = PointVector::dim;
    MPI_Win_create(data(), mysize*dim*sizeof(Atom), dim*sizeof(Atom), MPI_INFO_NULL, comm, &win);
}


DistPointVector::DistPointVector(const PointVector& mypoints, MPI_Comm comm)
    : PointVector(mypoints),
      comm(comm),
      mysize(mypoints.num_points())
{
    assert((PointVector::dim <= MAX_DIM));

    init_comm();
    init_offsets();
    init_window();
}

void DistPointVector::init_from_file(const char *fname, Index& total, size_t& disp, MPI_File *fh)
{
    int d;
    MPI_Offset filesize;

    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, fh);
    MPI_File_get_size(*fh, &filesize);

    if (!myrank)
    {
        MPI_File_read(*fh, &d, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&d, 1, MPI_INT, 0, comm);
    MPI_Bcast(&filesize, 1, MPI_OFFSET, 0, comm);

    assert((d <= MAX_DIM));
    PointVector::dim = d;

    disp = 4 * (d + 1);
    total = filesize / disp;

    assert((sizeof(Atom) == 4));
    assert((filesize % disp == 0));
}

DistPointVector::DistPointVector(const char* fname, MPI_Comm comm)
    : comm(comm)
{
    size_t disp;
    Index total, myleft;
    MPI_File fh;

    init_comm();
    init_from_file(fname, total, disp, &fh);
    int dim = PointVector::dim;

    mysize = total / nprocs;
    myleft = total % nprocs;

    if (myrank < myleft)
        mysize++;

    init_offsets();
    resize(mysize);

    MPI_Offset fileoffset = myoffset*disp;

    MPI_Datatype filetype;;
    MPI_Type_create_resized(MPI_POINT, 0, (MPI_Aint)disp, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, fileoffset+sizeof(int), MPI_POINT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read(fh, data(), mysize, MPI_POINT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);

    init_window();
}


DistPointVector::~DistPointVector()
{
    MPI_Type_free(&MPI_POINT);
    MPI_Win_free(&win);
}


PointVector DistPointVector::allgather(const IndexVector& myindices, IndexVector& indices) const
{
    /*
     * `myindices` contains local indices and `indices` will contain global indices on exit
     */

    int dim = PointVector::dim;
    int count = myindices.size();
    std::vector<int> disps(count);
    IndexVector send_indices = myindices;

    for (int i = 0; i < count; ++i)
    {
        disps[i] = myindices[i];
        send_indices[i] += myoffset;
    }

    MPI_Datatype MPI_INDEXED;
    MPI_Type_create_indexed_block(count, 1, disps.data(), MPI_POINT, &MPI_INDEXED);
    MPI_Type_commit(&MPI_INDEXED);

    std::vector<int> recvcounts(nprocs), rdispls(nprocs);
    recvcounts[myrank] = count;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
    int totrecv = recvcounts.back() + rdispls.back();

    indices.resize(totrecv);
    AtomVector recvbuf(totrecv*dim);

    MPI_Request reqs[2];

    MPI_Iallgatherv(send_indices.data(), count, MPI_INDEX, indices.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm, &reqs[0]);
    MPI_Iallgatherv(data(), 1, MPI_INDEXED, recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_POINT, comm, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    MPI_Type_free(&MPI_INDEXED);

    return PointVector(recvbuf, dim);
}

PointVector DistPointVector::allgather(const IndexVector& indices) const
{
    IndexVector myindices, dummy;

    for (Index id : indices)
        if (myoffset <= id && id < myoffset+mysize)
            myindices.push_back(id-myoffset);

    return allgather(myindices, dummy);
}

PointVector DistPointVector::gather_rma(const IndexVector& indices) const
{
    int dim = PointVector::dim;
    std::vector<MPI_Datatype> datatypes(nprocs);
    std::vector<std::vector<int>> displs(nprocs);

    for (Index id : indices)
    {
        int owner = point_owner(id);
        displs[owner].push_back(id-offsets[owner]);
    }

    for (int i = 0; i < nprocs; ++i)
    {
        int count = displs[i].size();
        MPI_Type_create_indexed_block(count, 1, displs[i].data(), MPI_POINT, &datatypes[i]);
        MPI_Type_commit(&datatypes[i]);
    }

    AtomVector recvbuf(indices.size()*dim);
    Atom *ptr = recvbuf.data();

    MPI_Win_lock_all(0,win);

    for (int i = 0; i < nprocs; ++i)
    {
        int count = displs[i].size();
        MPI_Get(ptr, count, MPI_POINT, i, 0, 1, datatypes[i], win);
        ptr += (count*dim);
    }

    MPI_Win_unlock_all(win);

    for (int i = 0; i < nprocs; ++i)
    {
        MPI_Type_free(&datatypes[i]);
    }

    return PointVector(recvbuf, dim);
}

struct BruteForceQuery
{
    Index operator()(const PointVector& mypoints, const PointVector& curpoints, Index curoffset, Index myoffset, Real radius, DistGraph& graph)
    {
        Index mysize = mypoints.num_points();
        Index cursize = curpoints.num_points();
        Index edges_found = 0;

        for (Index j = 0; j < cursize; ++j)
        {
            IndexVector neighs;

            for (Index i = 0; i < mysize; ++i)
                if (mypoints.distance(i, curpoints[j]) <= radius)
                    neighs.push_back(i+myoffset);

            graph.add_neighbors(j+curoffset, neighs);
            edges_found += neighs.size();
        }

        return edges_found;
    }
};

void DistPointVector::brute_force_systolic(Real radius, DistGraph& graph, int verbosity) const
{
    BruteForceQuery query;
    systolic(radius, query, graph, verbosity);
}


struct CoverTreeQuery
{
    CoverTree tree;

    CoverTreeQuery(const PointVector& mypoints, Real cover, Index leaf_size)
    {
        tree.build(mypoints, cover, leaf_size);
    }

    Index operator()(const PointVector& mypoints, const PointVector& curpoints, Index curoffset, Index myoffset, Real radius, DistGraph& graph)
    {
        Index cursize = curpoints.num_points();
        Index edges_found = 0;

        for (Index j = 0; j < cursize; ++j)
        {
            IndexVector neighs;
            edges_found += tree.radius_query(mypoints, curpoints[j], radius, neighs);
            graph.add_neighbors(j+curoffset, neighs, myoffset);
        }

        return edges_found;
    }
};

void DistPointVector::cover_tree_systolic(Real radius, Real cover, Index leaf_size, DistGraph& graph, int verbosity) const
{
    double mytime, maxtime;

    mytime = -MPI_Wtime();
    CoverTreeQuery query(*this, cover, leaf_size);
    mytime += MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,myrank=%d,time=%.3f] built local tree [points=%lld,vertices=%lld]\n", myrank, mytime, mysize, query.tree.num_vertices());
        fflush(stdout);
    }

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) printf("[v1,time=%.3f] built cover trees\n", maxtime);
        fflush(stdout);
    }

    systolic(radius, query, graph, verbosity);
}

template <class Query>
void DistPointVector::systolic(Real radius, Query& query, DistGraph& graph, int verbosity) const
{
    MPI_Request reqs[2];

    int next = (myrank+1)%nprocs;
    int prev = (myrank-1+nprocs)%nprocs;
    int cur = myrank;
    int dim = PointVector::dim;

    PointVector curpoints(*this);
    PointVector nextpoints;

    MPI_Barrier(comm);

    double mytime, tottime, maxtime;
    Index edges = 0;

    tottime = -MPI_Wtime();

    for (int step = 0; step < nprocs; ++step)
    {
        mytime = -MPI_Wtime();

        int sendcount = get_rank_size(cur);
        int recvcount = get_rank_size((cur+1)%nprocs);

        nextpoints.resize(recvcount, dim);

        MPI_Irecv(nextpoints.data(), recvcount, MPI_POINT, next, myrank, comm, &reqs[0]);
        MPI_Isend(curpoints.data(), sendcount, MPI_POINT, prev, prev, comm, &reqs[1]);

        Index cursize = get_rank_size(cur);
        Index curoffset = get_rank_offset(cur);
        Index found = query(*this, curpoints, curoffset, myoffset, radius, graph);
        edges += found;

        mytime += MPI_Wtime();

        if (verbosity >= 3)
        {
            printf("[v3,step=%d,rank=%d,time=%.3f] computed [%lld..%lld] vs [%lld..%lld] [edges=%lld]\n", step, myrank, mytime, myoffset, myoffset+mysize-1, curoffset, curoffset+cursize-1, found);
            fflush(stdout);
        }

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        cur = (cur+1)%nprocs;
        curpoints.swap(nextpoints);
    }

    tottime += MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,rank=%d,time=%.3f] computed [%lld..%lld] vs all [edges=%lld]\n", myrank, tottime, myoffset, myoffset+mysize-1, edges);
        fflush(stdout);
    }
}

