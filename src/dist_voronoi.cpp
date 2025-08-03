#include "dist_voronoi.h"
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <string.h>

void DistVoronoi::mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype)
{
    GlobalPoint *in = (GlobalPoint *)_in;
    GlobalPoint *inout = (GlobalPoint *)_inout;

    for (int i = 0; i < *len; ++i)
        if (in[i].dist > inout[i].dist)
            inout[i] = in[i];
}

DistVoronoi::DistVoronoi(const DistPointVector& points)
    : DistPointVector(points),
      centers(dim),
      cells(mysize, 0),
      dists(mysize, std::numeric_limits<Real>::max())
{
    GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, dim);
    MPI_Op_create(&mpi_argmax, 0, &MPI_ARGMAX);

    if (!myrank)
    {
        next_center.set_point(*this, 0);
        next_center.id = 0;
        next_center.dist = std::numeric_limits<Real>::max();
        next_center.cell = 0;
    }

    MPI_Bcast(&next_center, 1, MPI_GLOBAL_POINT, 0, comm);
}

DistVoronoi::~DistVoronoi()
{
    MPI_Type_free(&MPI_GLOBAL_POINT);
    MPI_Op_free(&MPI_ARGMAX);
}

void DistVoronoi::add_next_center()
{
    Index cell = num_centers();
    centers.push_back(next_center.p);
    centerids.push_back(next_center.id);

    next_center.dist = 0;

    for (Index i = 0; i < mysize; ++i)
    {
        Real dist = PointVector::distance(i, next_center.p);

        if (dist < dists[i])
        {
            dists[i] = dist;
            cells[i] = cell;
        }

        if (dists[i] > next_center.dist)
        {
            next_center.id = myoffset+i;
            next_center.dist = dists[i];
        }
    }

    next_center.set_point(*this, next_center.id-myoffset);

    MPI_Allreduce(MPI_IN_PLACE, &next_center, 1, MPI_GLOBAL_POINT, MPI_ARGMAX, comm);
}

void DistVoronoi::add_next_centers(Index count)
{
    centers.reserve(centers.num_points() + count);

    for (Index i = 0; i < count; ++i)
        add_next_center();
}

void DistVoronoi::gather_local_cell_ids(IndexVector& mycellids, IndexVector& ptrs) const
{
    Index m = num_centers();
    IndexVector w(m, 0);
    ptrs.resize(m+1);

    for (Index i = 0; i < mysize; ++i)
        w[cells[i]]++;

    Index nz = 0;

    for (Index i = 0; i < m; ++i)
    {
        ptrs[i] = nz;
        nz += w[i];
        w[i] = ptrs[i];
    }

    assert((nz == mysize));
    ptrs[m] = mysize;

    mycellids.resize(mysize);

    for (Index i = 0; i < mysize; ++i)
    {
        mycellids[w[cells[i]]++] = i;
    }
}

void DistVoronoi::gather_local_ghost_ids(Real radius, IndexVector& myghostids, IndexVector& ptrs) const
{
    Index m = num_centers();
    IndexVector w(m, 0);
    ptrs.resize(m+1);

    CoverTree reptree;
    reptree.build(centers, 1.3, 1);

    IndexPairVector ghostpairs;

    for (Index i = 0; i < mysize; ++i)
    {
        IndexVector ghostcells;
        reptree.radius_query(centers, (*this)[i], dists[i] + 2*radius, ghostcells);

        for (Index ghostcell : ghostcells)
            if (cells[i] != ghostcell)
            {
                ghostpairs.emplace_back(ghostcell, i);
                w[ghostcell]++;
            }
    }

    Index nz = 0;

    for (Index i = 0; i < m; ++i)
    {
        ptrs[i] = nz;
        nz += w[i];
        w[i] = ptrs[i];
    }

    ptrs[m] = nz;
    myghostids.resize(nz);

    for (const auto& [ghostcell, id] : ghostpairs)
    {
        myghostids[w[ghostcell]++] = id;
    }
}

void DistVoronoi::load_alltoall_outbufs(const IndexVector& ids, const IndexVector& ptrs, const std::vector<int>& dests, GlobalPointVector& sendbuf, std::vector<int>& sendcounts, std::vector<int>& sdispls) const
{
    Index m = num_centers();
    assert((ptrs.size() == m+1));
    assert((dests.size() == m));

    sendbuf.clear();
    sendcounts.resize(nprocs,0), sdispls.resize(nprocs);

    Index totsend = ids.size();

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];
        sendcounts[dest] += (ptrs[i+1]-ptrs[i]);
    }

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0));
    sendbuf.resize(totsend);

    std::vector<int> sendptrs = sdispls;

    IndexVector rank_cell_map(m), rank_cell_counts(nprocs,0);

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];
        rank_cell_map[i] = rank_cell_counts[dest];
        rank_cell_counts[dest]++;

        for (Index ptr = ptrs[i]; ptr < ptrs[i+1]; ++ptr)
        {
            Index id = ids[ptr];
            Index loc = sendptrs[dest]++;

            sendbuf[loc].set_point(*this, id);
            sendbuf[loc].id = id+myoffset;
            sendbuf[loc].cell = rank_cell_map[i];
            sendbuf[loc].dist = dists[id];
        }
    }
}

void DistVoronoi::get_stats(Index& mincellsize, Index& maxcellsize, int root) const
{
    int m = num_centers();
    IndexVector cellsizes(m, 0);

    for (Index cell : cells) cellsizes[cell]++;

    const void *sendbuf = myrank == root? MPI_IN_PLACE : cellsizes.data();

    MPI_Reduce(sendbuf, cellsizes.data(), m, MPI_INDEX, MPI_SUM, root, comm);

    if (myrank == root)
    {
        mincellsize = *std::min_element(cellsizes.begin(), cellsizes.end());
        maxcellsize = *std::max_element(cellsizes.begin(), cellsizes.end());
    }
}

Index DistVoronoi::compute_static_cyclic_assignments(std::vector<int>& dests, IndexVector& mycells) const
{
    Index s = 0;
    Index m = num_centers();

    dests.resize(m);
    mycells.clear();

    for (Index i = 0; i < m; ++i)
    {
        dests[i] = i % nprocs;

        if (dests[i] == myrank)
        {
            mycells.push_back(i);
            s++;
        }
    }

    return s;
}

Index DistVoronoi::compute_multiway_number_partitioning_assignments(std::vector<int>& dests, IndexVector& mycells) const
{
    Index s = 0;
    Index m = num_centers();

    dests.resize(m);
    mycells.clear();

    IndexVector cellsizes(m, 0);
    for (Index cell : cells) cellsizes[cell]++;

    MPI_Allreduce(MPI_IN_PLACE, cellsizes.data(), (int)m, MPI_INDEX, MPI_SUM, comm);

    IndexPairVector pairs;

    for (Index i = 0; i < m; ++i)
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

    return s;
}

void DistVoronoi::gather_assigned_points(const std::vector<int>& dests, Real radius, std::vector<PointVector>& my_cell_points, std::vector<IndexVector>& my_cell_indices, IndexVector& my_query_sizes, int verbosity) const
{
    double mytime, maxtime;

    IndexVector mycellids, myghostids, mycellptrs, myghostptrs;
    std::vector<int> cell_sendcounts, cell_recvcounts, cell_sdispls, cell_rdispls;
    std::vector<int> ghost_sendcounts, ghost_recvcounts, ghost_sdispls, ghost_rdispls;
    GlobalPointVector cell_sendbuf, cell_recvbuf;
    GlobalPointVector ghost_sendbuf, ghost_recvbuf;

    /*
     * Gather cell points and find ghost points
     */

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    gather_local_cell_ids(mycellids, mycellptrs);
    gather_local_ghost_ids(radius, myghostids, myghostptrs);
    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        Index my_num_ghosts = myghostids.size(), num_ghosts;

        if (verbosity >= 2) { printf("[v2,rank=%d,time=%.3f] found %lld ghost points [cell_points=%lu]\n", myrank, mytime, my_num_ghosts, mycellids.size()); fflush(stdout); }

        MPI_Reduce(&my_num_ghosts, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) printf("[v1,time=%.3f] found %lld ghost points\n", maxtime, num_ghosts);
    }

    MPI_Request reqs[2];

    load_alltoall_outbufs(mycellids, mycellptrs, dests, cell_sendbuf, cell_sendcounts, cell_sdispls);
    load_alltoall_outbufs(myghostids, myghostptrs, dests, ghost_sendbuf, ghost_sendcounts, ghost_sdispls);
    global_point_alltoall(cell_sendbuf, cell_sendcounts, cell_sdispls, cell_recvbuf, &reqs[0]);
    global_point_alltoall(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, ghost_recvbuf, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    Index s = my_cell_points.size();
    assert((s == my_query_sizes.size()));

    std::fill(my_query_sizes.begin(), my_query_sizes.end(), 0);
    IndexVector my_vector_sizes(s, 0);

    for (const auto& [p, id, cell, dist] : cell_recvbuf) { my_query_sizes[cell]++; my_vector_sizes[cell]++; }
    for (const auto& [p, id, cell, dist] : ghost_recvbuf) { my_vector_sizes[cell]++; }

    for (Index i = 0; i < s; ++i)
    {
        my_cell_points[i].clear();
        my_cell_points[i].reserve(my_vector_sizes[i]);
        my_cell_indices[i].reserve(my_vector_sizes[i]);
    }

    /* std::sort(cell_recvbuf.begin(), cell_recvbuf.end(), [](const auto& lhs, const auto& rhs) { return lhs.dist < rhs.dist; }); */

    for (const auto& p : cell_recvbuf) { my_cell_points[p.cell].push_back(p.p); my_cell_indices[p.cell].push_back(p.id); }
    for (const auto& p : ghost_recvbuf) { my_cell_points[p.cell].push_back(p.p); my_cell_indices[p.cell].push_back(p.id); }
}

void DistVoronoi::global_point_alltoall(const GlobalPointVector& sendbuf, const std::vector<int>& sendcounts, const std::vector<int>& sdispls, GlobalPointVector& recvbuf, MPI_Request *request) const
{
    std::vector<int> recvcounts(nprocs), rdispls(nprocs);

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    recvbuf.resize(recvcounts.back()+rdispls.back());

    MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_GLOBAL_POINT,
                   recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_GLOBAL_POINT, comm, request);
}
