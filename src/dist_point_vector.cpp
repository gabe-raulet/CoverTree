#include "dist_point_vector.h"
#include "cover_tree.h"
#include "work_stealer.h"
#include "timer.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <filesystem>
#include <limits>

DistPointVector::DistPointVector(const char *fname, MPI_Comm comm)
    : comm(comm)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_File fh;
    MPI_Aint extent;
    MPI_Offset filesize;
    Index total, myleft;

    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    if (!myrank)
    {
        MPI_File_get_size(fh, &filesize);
        MPI_File_read(fh, &dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
    MPI_Bcast(&filesize, 1, MPI_OFFSET, 0, comm);

    assert((dim <= MAX_DIM));

    extent = 4 * (dim + 1);
    total = filesize / extent;

    assert((sizeof(Atom) == 4));
    assert((filesize % extent == 0));

    mysize = total / nprocs;
    myleft = total % nprocs;

    if (myrank < myleft)
        mysize++;

    IndexVector sizes(nprocs);
    sizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, sizes.data(), 1, MPI_INDEX, comm);

    offsets.resize(nprocs);
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), (Index)0);

    totsize = offsets.back() + sizes.back();
    myoffset = offsets[myrank];

    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    resize(mysize);

    MPI_Datatype filetype;
    MPI_Type_create_resized(MPI_POINT, 0, extent, &filetype);
    MPI_Type_commit(&filetype);

    MPI_Offset filedisp = myoffset*extent+sizeof(int);
    MPI_File_set_view(fh, filedisp, MPI_POINT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read(fh, data(), (int)mysize, MPI_POINT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);

    MPI_Win_create(data(), mysize*dim*sizeof(Atom), dim*sizeof(Atom), MPI_INFO_NULL, comm, &win);
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

PointVector DistPointVector::gather_rma_block(int target_rank) const
{
    int dim = PointVector::dim;

    int target_size = get_rank_size(target_rank);

    AtomVector recvbuf(target_size*dim);

    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
    MPI_Get(recvbuf.data(), target_size, MPI_POINT, target_rank, 0, target_size, MPI_POINT, win);
    MPI_Win_unlock(target_rank, win);

    return PointVector(recvbuf, dim);
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
    Timer timer(comm);

    timer.start();
    CoverTreeQuery query(*this, cover, leaf_size);
    timer.stop();

    if (verbosity >= 2)
    {
        printf("[v2,%s] built local tree [points=%lld,vertices=%lld]\n", timer.myrepr().c_str(), mysize, query.tree.num_vertices());
        fflush(stdout);
    }

    timer.wait();

    if (verbosity >= 1)
    {
        if (!myrank) printf("[v1,time=%s] built cover trees\n", timer.repr().c_str());
        fflush(stdout);
    }

    systolic(radius, query, graph, verbosity);
}

void DistPointVector::cover_tree_rma(Real radius, Real cover, Index leaf_size, DistGraph& graph, int verbosity) const
{
    double t;

    Timer fulltimer(comm), timer(comm);

    fulltimer.start();
    t = -MPI_Wtime();
    CoverTreeQuery query(*this, cover, leaf_size);
    t += MPI_Wtime();
    fulltimer.stop();

    my_comp_time += t;

    if (verbosity >= 2)
    {
        printf("[v2,%s] built local tree [points=%lld,vertices=%lld]\n", fulltimer.myrepr().c_str(), mysize, query.tree.num_vertices());
        fflush(stdout);
    }

    fulltimer.wait();

    if (verbosity >= 1)
    {
        if (!myrank) printf("[v1,time=%s] built cover trees\n", fulltimer.repr().c_str());
        fflush(stdout);
    }

    fulltimer.start();

    Index edges = 0;

    int target_rank = myrank+1;

    for (int step = 0; step < nprocs; ++step, ++target_rank)
    {
        target_rank = target_rank % nprocs;

        Index found;
        timer.start();

        Index target_offset = get_rank_offset(target_rank);

        if (target_rank == myrank)
        {
            t = -MPI_Wtime();
            found = query(*this, *this, myoffset, myoffset, radius, graph);
            t += MPI_Wtime();
            my_comp_time += t;
        }
        else
        {
            t = -MPI_Wtime();
            PointVector target_points = gather_rma_block(target_rank);
            t += MPI_Wtime();
            my_comm_time += t;

            t = -MPI_Wtime();
            found = query(*this, target_points, target_offset, myoffset, radius, graph);
            t += MPI_Wtime();
            my_comp_time += t;
        }

        edges += found;
        timer.stop();

        if (verbosity >= 3)
        {
            printf("[v3,target=%d,%s] computed [%lld..%lld] vs [%lld..%lld] [edges=%lld]\n", target_rank, timer.myrepr().c_str(), myoffset, myoffset+mysize-1, target_offset, target_offset+get_rank_size(target_rank)-1, found);
            fflush(stdout);
        }
    }

    fulltimer.stop();

    if (verbosity >= 2)
    {
        printf("[v2,%s] computed [%lld..%lld] vs all [edges=%lld]\n", fulltimer.myrepr().c_str(), myoffset, myoffset+mysize-1, edges);
        fflush(stdout);
    }
}

template <class Query>
void DistPointVector::systolic(Real radius, Query& query, DistGraph& graph, int verbosity) const
{
    double t;

    MPI_Request reqs[2];

    t = -MPI_Wtime();

    int next = (myrank+1)%nprocs;
    int prev = (myrank-1+nprocs)%nprocs;
    int cur = myrank;
    int dim = PointVector::dim;

    PointVector curpoints(*this);
    PointVector nextpoints;

    t += MPI_Wtime();
    my_comp_time += t;

    MPI_Barrier(comm);

    Timer timer(comm), fulltimer(comm);
    Index edges = 0;

    fulltimer.start();

    for (int step = 0; step < nprocs; ++step)
    {
        timer.start();

        t = -MPI_Wtime();

        int sendcount = get_rank_size(cur);
        int recvcount = get_rank_size((cur+1)%nprocs);

        nextpoints.resize(recvcount, dim);

        t += MPI_Wtime();
        my_comp_time += t;

        t = -MPI_Wtime();
        MPI_Irecv(nextpoints.data(), recvcount, MPI_POINT, next, myrank, comm, &reqs[0]);
        MPI_Isend(curpoints.data(), sendcount, MPI_POINT, prev, prev, comm, &reqs[1]);
        t += MPI_Wtime();
        my_comm_time += t;

        t = -MPI_Wtime();
        Index cursize = get_rank_size(cur);
        Index curoffset = get_rank_offset(cur);
        Index found = query(*this, curpoints, curoffset, myoffset, radius, graph);
        edges += found;
        t += MPI_Wtime();
        my_comp_time += t;

        t = -MPI_Wtime();

        timer.stop();

        if (verbosity >= 3)
        {
            printf("[v3,step=%d,%s] computed [%lld..%lld] vs [%lld..%lld] [edges=%lld]\n", step, timer.myrepr().c_str(), myoffset, myoffset+mysize-1, curoffset, curoffset+cursize-1, found);
            fflush(stdout);
        }

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        t += MPI_Wtime();
        my_comm_time += t;

        timer.wait();

        cur = (cur+1)%nprocs;
        curpoints.swap(nextpoints);
    }

    fulltimer.stop();

    if (verbosity >= 2)
    {
        printf("[v2,%s] computed [%lld..%lld] vs all [edges=%lld]\n", fulltimer.myrepr().c_str(), myoffset, myoffset+mysize-1, edges);
        fflush(stdout);
    }

    fulltimer.wait();
}

struct GlobalPoint { Atom p[MAX_DIM]; Index i; Real r; };

void mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype)
{
    GlobalPoint *in = (GlobalPoint *)_in;
    GlobalPoint *inout = (GlobalPoint *)_inout;

    for (int i = 0; i < *len; ++i)
        if (in[i].r > inout[i].r)
            inout[i] = in[i];
}

void DistPointVector::build_voronoi_diagram(Index num_centers, PointVector& centers, IndexVector& centerids, IndexVector& cells, RealVector& dists, int verbosity) const
{
    double t;

    t = -MPI_Wtime();

    centers.clear();
    centerids.clear();

    centers.reserve(num_centers, dim);
    centerids.reserve(num_centers);

    GlobalPoint next_center;

    cells.resize(mysize, 0);
    dists.resize(mysize, std::numeric_limits<Real>::max());

    MPI_Op MPI_ARGMAX;
    MPI_Datatype MPI_GLOBAL_POINT;

    int blklens[3] = {dim,1,1};
    MPI_Aint disps[3] = {offsetof(GlobalPoint, p), offsetof(GlobalPoint, i), offsetof(GlobalPoint, r)};
    MPI_Datatype types[3] = {MPI_ATOM, MPI_INDEX, MPI_REAL};
    MPI_Type_create_struct(3, blklens, disps, types, &MPI_GLOBAL_POINT);
    MPI_Type_commit(&MPI_GLOBAL_POINT);
    MPI_Op_create(&mpi_argmax, 0, &MPI_ARGMAX);

    t += MPI_Wtime();
    my_comp_time += t;

    Timer timer(comm);

    MPI_Barrier(comm);
    timer.start();

    t = -MPI_Wtime();

    if (!myrank)
    {
        std::copy(begin(0), end(0), next_center.p);
        next_center.i = 0;
        next_center.r = std::numeric_limits<Real>::max();
    }

    t += MPI_Wtime();
    my_comp_time += t;

    t = -MPI_Wtime();
    MPI_Bcast(&next_center, 1, MPI_GLOBAL_POINT, 0, comm);
    t += MPI_Wtime();
    my_comm_time += t;

    for (Index cell = 0; cell < num_centers; ++cell)
    {
        t = -MPI_Wtime();

        centers.push_back(next_center.p);
        centerids.push_back(next_center.i);

        next_center.r = 0;

        for (Index i = 0; i < mysize; ++i)
        {
            Real dist = distance(i, next_center.p);

            if (dist < dists[i])
            {
                dists[i] = dist;
                cells[i] = cell;
            }

            if (dists[i] > next_center.r)
            {
                next_center.i = i;
                next_center.r = dists[i];
            }
        }

        std::copy(begin(next_center.i), end(next_center.i), next_center.p);
        next_center.i += myoffset;

        t += MPI_Wtime();
        my_comp_time += t;

        t = -MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &next_center, 1, MPI_GLOBAL_POINT, MPI_ARGMAX, comm);
        t += MPI_Wtime();
        my_comm_time += t;
    }

    MPI_Type_free(&MPI_GLOBAL_POINT);
    MPI_Op_free(&MPI_ARGMAX);

    timer.stop();
    timer.wait();

    IndexVector cellsizes(num_centers, 0);
    for (Index cell : cells) cellsizes[cell]++;

    const void *sendbuf = myrank == 0? MPI_IN_PLACE : cellsizes.data();
    MPI_Reduce(sendbuf, cellsizes.data(), (int)num_centers, MPI_INDEX, MPI_SUM, 0, comm);

    if (verbosity >= 1 && !myrank)
    {
        Index mincellsize = *std::min_element(cellsizes.begin(), cellsizes.end());
        Index maxcellsize = *std::max_element(cellsizes.begin(), cellsizes.end());
        printf("[v1,%s] found %lld centers [separation=%.3f,minsize=%lld,maxsize=%lld,avgsize=%.3f]\n", timer.repr().c_str(), num_centers, next_center.r, mincellsize, maxcellsize, (totsize+0.0)/num_centers);
    }
}

void DistPointVector::find_ghost_points(Real radius, Real cover, const PointVector& centers, const IndexVector& cells, const RealVector& dists, std::vector<IndexVector>& mycellids, std::vector<IndexVector>& myghostids, int verbosity) const
{
    double t;

    Timer timer(comm);
    timer.start();

    t = -MPI_Wtime();

    CoverTree reptree;
    reptree.build(centers, cover, 1);
    Index total = 0;

    Index num_centers = centers.num_points();

    myghostids.clear(), mycellids.clear();
    myghostids.resize(num_centers), mycellids.resize(num_centers);

    for (Index i = 0; i < mysize; ++i)
    {
        IndexVector ghostcells;
        reptree.radius_query(centers, (*this)[i], dists[i] + 2*radius, ghostcells);

        for (Index ghostcell : ghostcells)
            if (cells[i] != ghostcell)
            {
                myghostids[ghostcell].push_back(i);
                total++;
            }

        mycellids[cells[i]].push_back(i);
    }

    t += MPI_Wtime();
    my_comp_time += t;

    t = -MPI_Wtime();

    timer.stop();

    if (verbosity >= 2)
    {
        printf("[v2,%s] found %lld local ghost points\n", timer.myrepr().c_str(), total);
        fflush(stdout);
    }

    timer.wait();

    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1)
    {
        Index num_ghosts;
        MPI_Reduce(&total, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) printf("[v1,%s] found %lld total ghost points\n", timer.repr().c_str(), num_ghosts);

        fflush(stdout);
    }
}

Index DistPointVector::compute_assignments(Index num_centers, const IndexVector& cells, const char *tree_assignment, std::vector<int>& dests, IndexVector& mycells, int verbosity) const
{
    double t;

    Timer timer(comm);
    timer.start();

    Index s = 0;
    dests.resize(num_centers);
    mycells.clear();

    if (!strcmp(tree_assignment, "static"))
    {
        t = -MPI_Wtime();

        for (Index i = 0; i < num_centers; ++i)
        {
            dests[i] = i % nprocs;

            if (dests[i] == myrank)
            {
                mycells.push_back(i);
                s++;
            }
        }

        t += MPI_Wtime();
        my_comp_time += t;
    }
    else if (!strcmp(tree_assignment, "multiway"))
    {
        t = -MPI_Wtime();
        IndexVector cellsizes(num_centers, 0);
        for (Index cell : cells) cellsizes[cell]++;
        t += MPI_Wtime();
        my_comp_time += t;

        t = -MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, cellsizes.data(), (int)num_centers, MPI_INDEX, MPI_SUM, comm);
        t += MPI_Wtime();
        my_comm_time += t;

        t = -MPI_Wtime();
        IndexPairVector pairs;

        for (Index i = 0; i < num_centers; ++i)
        {
            pairs.emplace_back(cellsizes[i], i);
        }

        std::sort(pairs.rbegin(), pairs.rend());

        IndexVector bins(nprocs, 0);

        for (const auto& [size, cell] : pairs)
        {
            int rank = std::min_element(bins.begin(), bins.end()) - bins.begin();
            bins[rank] += size;
            dests[cell] = rank;

            if (rank == myrank)
            {
                mycells.push_back(cell);
                s++;
            }
        }

        t += MPI_Wtime();
        my_comp_time += t;
    }

    timer.stop();
    timer.wait();

    if (verbosity >= 1)
    {
        if (!myrank) printf("[v1,%s] computed assignments [method=%s]\n", timer.repr().c_str(), tree_assignment);
        fflush(stdout);
    }

    return s;
}

void DistPointVector::global_point_alltoall(const std::vector<IndexVector>& ids, const std::vector<int>& dests, std::vector<PointVector>& my_cell_points, std::vector<IndexVector>& my_cell_indices, IndexVector& my_sizes, int verbosity) const
{
    double t;

    Timer timer(comm);
    timer.start();

    t = -MPI_Wtime();

    Index m = dests.size();
    std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);

    assert((ids.size() == m));
    assert((dests.size() == m));

    Index totsend = 0;
    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];
        sendcounts[dest] += ids[i].size();
        totsend += ids[i].size();
    }

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0));

    AtomVector sendbuf_atoms(totsend*dim), recvbuf_atoms;
    IndexVector sendbuf_ids(totsend), recvbuf_ids;
    IndexVector sendbuf_cells(totsend), recvbuf_cells;

    std::vector<int> sendptrs = sdispls;

    IndexVector rank_cell_counts(nprocs,0);

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];

        for (Index id : ids[i])
        {
            Index loc = sendptrs[dest]++;

            std::copy(begin(id), end(id), sendbuf_atoms.begin() + loc*dim);
            sendbuf_ids[loc] = id+myoffset;
            sendbuf_cells[loc] = rank_cell_counts[dest];
        }

        rank_cell_counts[dest]++;
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    Index totrecv = recvcounts.back() + rdispls.back();

    recvbuf_atoms.resize(totrecv*dim);
    recvbuf_ids.resize(totrecv);
    recvbuf_cells.resize(totrecv);

    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    MPI_Request reqs[3];

    MPI_Ialltoallv(sendbuf_ids.data(), sendcounts.data(), sdispls.data(), MPI_INDEX,
                   recvbuf_ids.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm, &reqs[0]);

    MPI_Ialltoallv(sendbuf_cells.data(), sendcounts.data(), sdispls.data(), MPI_INDEX,
                   recvbuf_cells.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm, &reqs[1]);

    MPI_Ialltoallv(sendbuf_atoms.data(), sendcounts.data(), sdispls.data(), MPI_POINT,
                   recvbuf_atoms.data(), recvcounts.data(), rdispls.data(), MPI_POINT, comm, &reqs[2]);

    MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);

    Index s = my_cell_points.size();

    for (Index i = 0; i < totrecv; ++i)
    {
        my_sizes[recvbuf_cells[i]]++;
    }

    for (Index i = 0; i < s; ++i)
    {
        my_cell_points[i].reserve(my_cell_points[i].num_points() + my_sizes[i]);
        my_cell_indices[i].reserve(my_cell_indices[i].size() + my_sizes[i]);
    }

    for (Index i = 0; i < totrecv; ++i)
    {
        const Atom *pt = &recvbuf_atoms[i*dim];
        Index cell = recvbuf_cells[i];
        Index id = recvbuf_ids[i];

        my_cell_points[cell].push_back(pt);
        my_cell_indices[cell].push_back(id);
    }

    t += MPI_Wtime();
    my_comm_time += t;

    timer.stop();

    t = -MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,%s] received %lld points alltoall\n", timer.myrepr().c_str(), totrecv);
        fflush(stdout);
    }

    timer.wait();
    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1)
    {
        Index totcomm;
        MPI_Reduce(&totrecv, &totcomm, 1, MPI_INDEX, MPI_SUM, 0, comm);
        if (!myrank) printf("[v1,%s] communicated %lld points alltoall\n", timer.repr().c_str(), totcomm);
        fflush(stdout);
    }
}

void DistPointVector::build_ghost_trees(std::vector<GhostTree>& mytrees, const std::vector<PointVector>& my_cell_points, const std::vector<IndexVector>& my_cell_indices, const IndexVector& my_query_sizes, const IndexVector& mycells, Real cover, Index leaf_size, int verbosity) const
{
    double t;

    Index s = my_cell_points.size();

    Index my_num_vertices = 0, num_vertices;

    Timer timer(comm);
    timer.start();

    t = -MPI_Wtime();

    for (Index i = 0; i < s; ++i)
    {
        CoverTree tree;
        tree.build(my_cell_points[i], cover, leaf_size);
        my_num_vertices += tree.num_vertices();
        mytrees.emplace_back(tree, my_cell_points[i], my_cell_indices[i], my_query_sizes[i], mycells[i]);
    }

    t += MPI_Wtime();
    my_comp_time += t;
    timer.stop();

    t = -MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,%s] built %lld local cover trees [num_vertices=%lld]\n", timer.myrepr().c_str(), s, my_num_vertices);
        fflush(stdout);
    }

    timer.wait();
    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1)
    {
        MPI_Reduce(&my_num_vertices, &num_vertices, 1, MPI_INDEX, MPI_SUM, 0, comm);
        if (!myrank) printf("[v1,%s] built all cover trees [num_vertices=%lld]\n", timer.repr().c_str(), num_vertices);
        fflush(stdout);
    }
}

void DistPointVector::build_cover_trees(std::vector<CoverTree>& mytrees, const std::vector<PointVector>& my_cell_points, const IndexVector& mycells, Real cover, Index leaf_size, int verbosity) const
{
    double t;

    mytrees.clear();
    Index s = my_cell_points.size();

    Index my_num_vertices = 0, num_vertices;

    Timer timer(comm);
    timer.start();

    t = -MPI_Wtime();

    for (Index i = 0; i < s; ++i)
    {
        mytrees.emplace_back();
        mytrees.back().build(my_cell_points[i], cover, leaf_size);
        my_num_vertices += mytrees.back().num_vertices();
    }

    t += MPI_Wtime();
    my_comp_time += t;
    timer.stop();

    t = -MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,%s] built %lld local cover trees [num_vertices=%lld]\n", timer.myrepr().c_str(), s, my_num_vertices);
        fflush(stdout);
    }

    timer.wait();

    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1)
    {
        MPI_Reduce(&my_num_vertices, &num_vertices, 1, MPI_INDEX, MPI_SUM, 0, comm);
        if (!myrank) printf("[v1,%s] built all cover trees [num_vertices=%lld]\n", timer.repr().c_str(), num_vertices);
        fflush(stdout);
    }

}

void DistPointVector::ghost_tree_voronoi(Real radius, Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, Index queries_per_tree, DistGraph& graph, int verbosity) const
{
    PointVector centers; /* size: num_centers */
    IndexVector centerids; /* size: num_centers */

    IndexVector cells; /* size: mysize(rank) */
    RealVector dists; /* size: mysize(rank) */

    build_voronoi_diagram(num_centers, centers, centerids, cells, dists, verbosity);

    std::vector<IndexVector> mycellids; /* size: num_centers */
    std::vector<IndexVector> myghostids; /* size: num_centers */

    find_ghost_points(radius, cover, centers, cells, dists, mycellids, myghostids, verbosity);

    std::vector<int> dests; /* size: num_centers */
    IndexVector mycells; /* size: s(rank) */

    Index s = compute_assignments(num_centers, cells, tree_assignment, dests, mycells, verbosity);

    IndexVector my_query_sizes(s,0), my_ghost_sizes(s,0);
    std::vector<PointVector> my_cell_points(s, PointVector(0,dim));
    std::vector<IndexVector> my_cell_indices(s);

    global_point_alltoall(mycellids, dests, my_cell_points, my_cell_indices, my_query_sizes, verbosity);
    global_point_alltoall(myghostids, dests, my_cell_points, my_cell_indices, my_ghost_sizes, verbosity);

    query_sizes.assign(my_query_sizes.begin(), my_query_sizes.end());
    ghost_sizes.assign(my_ghost_sizes.begin(), my_ghost_sizes.end());

    std::vector<GhostTree> mytrees;

    build_ghost_trees(mytrees, my_cell_points, my_cell_indices, my_query_sizes, mycells, cover, leaf_size, verbosity);
    find_neighbors_ghost(mytrees, radius, queries_per_tree, query_balancing, graph, verbosity);
}

void DistPointVector::cover_tree_voronoi(Real radius, Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, Index queries_per_tree, DistGraph& graph, int verbosity) const
{
    PointVector centers; /* size: num_centers */
    IndexVector centerids; /* size: num_centers */

    IndexVector cells; /* size: mysize(rank) */
    RealVector dists; /* size: mysize(rank) */

    build_voronoi_diagram(num_centers, centers, centerids, cells, dists, verbosity);

    std::vector<IndexVector> mycellids; /* size: num_centers */
    std::vector<IndexVector> myghostids; /* size: num_centers */

    find_ghost_points(radius, cover, centers, cells, dists, mycellids, myghostids, verbosity);

    std::vector<int> dests; /* size: num_centers */
    IndexVector mycells; /* size: s(rank) */

    Index s = compute_assignments(num_centers, cells, tree_assignment, dests, mycells, verbosity);

    IndexVector my_query_sizes(s,0), my_ghost_sizes(s,0);
    std::vector<PointVector> my_cell_points(s, PointVector(0,dim)), my_ghost_points(s, PointVector(0,dim));
    std::vector<IndexVector> my_cell_indices(s), my_ghost_indices(s);

    global_point_alltoall(mycellids, dests, my_cell_points, my_cell_indices, my_query_sizes, verbosity);
    global_point_alltoall(myghostids, dests, my_ghost_points, my_ghost_indices, my_ghost_sizes, verbosity);

    query_sizes.assign(my_query_sizes.begin(), my_query_sizes.end());
    ghost_sizes.assign(my_ghost_sizes.begin(), my_ghost_sizes.end());

    std::vector<CoverTree> mytrees;

    build_cover_trees(mytrees, my_cell_points, mycells, cover, leaf_size, verbosity);
    find_neighbors_cover(mytrees, my_cell_points, my_ghost_points, my_cell_indices, my_ghost_indices, radius, graph, verbosity);
}

void DistPointVector::find_neighbors_cover
(
    const std::vector<CoverTree>& mytrees,
    const std::vector<PointVector>& my_cell_points,
    const std::vector<PointVector>& my_ghost_points,
    const std::vector<IndexVector>& my_cell_indices,
    const std::vector<IndexVector>& my_ghost_indices,
    Real radius,
    DistGraph& graph,
    int verbosity
) const
{

    double t;

    Index s = mytrees.size();

    Index num_local_queries_made = 0;
    Index num_local_edges_found = 0;

    Timer timer(comm);
    timer.start();

    t = -MPI_Wtime();

    for (Index i = 0; i < s; ++i)
    {
        Index query_size = my_cell_indices[i].size();
        Index ghost_size = my_ghost_indices[i].size();

        for (Index query = 0; query < query_size; ++query)
        {
            IndexVector neighs;
            Index edges_found = mytrees[i].radius_query_indexed(my_cell_points[i], my_cell_indices[i], query, radius, neighs);
            graph.add_neighbors(my_cell_indices[i][query], neighs);

            num_local_edges_found += edges_found;
            num_local_queries_made++;
        }

        for (Index query = 0; query < ghost_size; ++query)
        {
            IndexVector neighs;
            Index edges_found = mytrees[i].radius_query_indexed(my_cell_points[i], my_cell_indices[i], my_ghost_points[i][query], radius, neighs);
            graph.add_neighbors(my_ghost_indices[i][query], neighs);

            num_local_edges_found += edges_found;
            num_local_queries_made++;
        }
    }

    t += MPI_Wtime();
    my_comp_time += t;

    timer.stop();

    t = -MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,%s] completed queries [queries=%lld,edges=%lld]\n", timer.myrepr().c_str(), num_local_queries_made, num_local_edges_found);
        fflush(stdout);
    }

    timer.wait();

    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1 && !myrank)
    {
        printf("[v1,%s] completed all queries\n", timer.repr().c_str());
    }
}

void DistPointVector::find_neighbors_ghost(const std::vector<GhostTree>& mytrees, Real radius, Index queries_per_tree, const char *query_balancing, DistGraph& graph, int verbosity) const
{
    double t;

    std::deque<GhostTree> myqueue(mytrees.begin(), mytrees.end());

    Index num_local_queries_made = 0;
    Index num_local_edges_found = 0;

    Timer timer(comm);
    MPI_Barrier(comm);

    auto make_tree_queries = [&](GhostTree& tree) -> Index
    {
        double t;

        t = -MPI_Wtime();

        Index queries_made;
        Index edges_found = tree.make_queries(queries_per_tree, radius, graph, queries_made);

        num_local_queries_made += queries_made;
        num_local_edges_found += edges_found;

        t += MPI_Wtime();

        if (verbosity >= 3)
        {
            printf("[v3,rank=%d,time=%.3f] queried ghost tree [id=%lld,made=%lld,left=%lld,found=%lld,calls=%lld]\n", myrank, t, tree.header.id, queries_made, tree.header.queries_remaining(), edges_found, tree.header.called);
            fflush(stdout);
        }

        return queries_made;
    };

    timer.start();

    if (!strcmp(query_balancing, "static"))
    {
        t = -MPI_Wtime();

        while (!myqueue.empty())
        {
            auto it = myqueue.begin();

            while (it != myqueue.end())
            {
                make_tree_queries(*it);

                if (it->finished())
                    it = myqueue.erase(it);
                else
                    it++;
            }
        }

        t += MPI_Wtime();
        my_comp_time += t;

        timer.stop();

        t = -MPI_Wtime();

        if (verbosity >= 2)
        {
            printf("[v2,%s] completed queries [queries=%lld,edges=%lld]\n", timer.myrepr().c_str(), num_local_queries_made, num_local_edges_found);
            fflush(stdout);
        }
    }
    else if (!strcmp(query_balancing, "steal"))
    {
        WorkStealer work_stealer(&myqueue, dim, comm);

        Index totsize = myqueue.size();
        MPI_Allreduce(MPI_IN_PLACE, &totsize, 1, MPI_INDEX, MPI_SUM, comm);

        double t;

        int flag;
        Index mycount = 0;
        Index sendbuf = 0, recvbuf;
        MPI_Request request;

        t = -MPI_Wtime();
        MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_INDEX, MPI_SUM, comm, &request);
        t += MPI_Wtime();
        my_allreduce_time += t;

        while (true)
        {
            work_stealer.poll_incoming_requests(my_poll_time, my_response_time);

            if (!myqueue.empty())
            {
                t = -MPI_Wtime();

                Index queries_made = make_tree_queries(myqueue.front());
                work_stealer.queries_remaining -= queries_made;

                if (myqueue.front().finished())
                {
                    myqueue.pop_front();
                    mycount++;
                }

                t += MPI_Wtime();
                my_steal_comp_time += t;
            }
            else
            {
                t = -MPI_Wtime();
                work_stealer.random_steal();
                t += MPI_Wtime();
                my_steal_time += t;
            }

            t = -MPI_Wtime();
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
            t += MPI_Wtime();
            my_allreduce_time += t;

            if (flag)
            {
                t = -MPI_Wtime();

                if (recvbuf >= totsize)
                    break;

                sendbuf = mycount;
                MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_INDEX, MPI_SUM, comm, &request);
                t += MPI_Wtime();
                my_allreduce_time += t;
            }
        }

        my_comp_time += my_steal_comp_time;
        my_comm_time += (my_steal_time + my_poll_time + my_response_time + my_allreduce_time);

        timer.stop();

        t = -MPI_Wtime();

        if (verbosity >= 2)
        {
            printf("[v2,%s] completed queries [queries=%lld,edges=%lld][comp=%.3f,steal=%.3f,poll=%.3f,resp=%.3f,term=%.3f][attempts=%lld,successes=%lld,serviced=%lld]\n", timer.myrepr().c_str(), num_local_queries_made, num_local_edges_found, my_steal_comp_time, my_steal_time, my_poll_time, my_response_time, my_allreduce_time, work_stealer.steal_attempts, work_stealer.steal_successes, work_stealer.steal_services);
            fflush(stdout);
        }
    }

    timer.wait();
    t += MPI_Wtime();
    my_idle_time += t;

    if (verbosity >= 1 && !myrank)
    {
        printf("[v1,%s] completed all queries\n", timer.repr().c_str());
    }

    fflush(stdout);
}
