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

Index DistVoronoi::gather_local_cell_ids(std::vector<IndexVector>& mycellids) const
{
    Index m = num_centers();

    mycellids.clear();
    mycellids.resize(m);

    for (Index i = 0; i < mysize; ++i)
    {
        mycellids[cells[i]].push_back(i);
    }

    return mysize;
}

Index DistVoronoi::gather_local_ghost_ids(Real radius, std::vector<IndexVector>& myghostids) const
{
    Index m = num_centers();

    myghostids.clear();
    myghostids.resize(m);

    CoverTree reptree;
    reptree.build(centers, 1.3, 1);
    Index total = 0;

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
    }

    return total;
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

    std::vector<IndexVector> mycellids, myghostids;

    /*
     * Gather cell points and find ghost points
     */

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    Index my_num_cell_ids = gather_local_cell_ids(mycellids);
    Index my_num_ghosts = gather_local_ghost_ids(radius, myghostids);
    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        Index num_ghosts;

        if (verbosity >= 2) { printf("[v2,rank=%d,time=%.3f] found %lld ghost points [cell_points=%lld]\n", myrank, mytime, my_num_ghosts, my_num_cell_ids); fflush(stdout); }

        MPI_Reduce(&my_num_ghosts, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) printf("[v1,time=%.3f] found %lld ghost points\n", maxtime, num_ghosts);
    }


    Index s = my_cell_points.size();
    assert((s == my_query_sizes.size()));
    IndexVector my_ghost_sizes(s, 0);

    global_point_alltoall(mycellids, dests, my_cell_points, my_cell_indices, my_query_sizes);
    global_point_alltoall(myghostids, dests, my_cell_points, my_cell_indices, my_ghost_sizes);
}

void DistVoronoi::global_point_alltoall(const std::vector<IndexVector>& ids, const std::vector<int>& dests, std::vector<PointVector>& my_cell_points, std::vector<IndexVector>& my_cell_indices, IndexVector& my_sizes) const
{
    int dim = num_dimensions();
    std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);

    Index m = num_centers();
    assert((ids.size() == m));
    assert((dests.size() == m));

    Index totsend = 0;;
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

    IndexVector rank_cell_map(m), rank_cell_counts(nprocs,0);

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];
        rank_cell_map[i] = rank_cell_counts[dest];
        rank_cell_counts[dest]++;

        for (Index id : ids[i])
        {
            Index loc = sendptrs[dest]++;

            std::copy(begin(id), end(id), sendbuf_atoms.begin() + loc*dim);
            sendbuf_ids[loc] = id+myoffset;
            sendbuf_cells[loc] = rank_cell_map[i];
        }
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

    MPI_Alltoallv(sendbuf_atoms.data(), sendcounts.data(), sdispls.data(), MPI_POINT,
                  recvbuf_atoms.data(), recvcounts.data(), rdispls.data(), MPI_POINT, comm);

    MPI_Alltoallv(sendbuf_ids.data(), sendcounts.data(), sdispls.data(), MPI_INDEX,
                  recvbuf_ids.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);

    MPI_Alltoallv(sendbuf_cells.data(), sendcounts.data(), sdispls.data(), MPI_INDEX,
                  recvbuf_cells.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);

    MPI_Type_free(&MPI_POINT);

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
}
