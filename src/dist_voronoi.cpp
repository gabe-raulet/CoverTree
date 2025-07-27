#include "dist_voronoi.h"
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

DistVoronoi::DistVoronoi(const PointVector& mypoints, Index global_seed, MPI_Comm comm)
    : centers(mypoints.num_dimensions()),
      mypoints(mypoints),
      cells(mypoints.num_points(), 0),
      dists(mypoints.num_points(), std::numeric_limits<Real>::max()),
      comm(comm),
      mysize(mypoints.num_points())
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Allreduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, comm);
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);

    if (!myrank) myoffset = 0;

    GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, mypoints.num_dimensions());
    MPI_Op_create(&mpi_argmax, 0, &MPI_ARGMAX);

    if (myoffset <= global_seed && global_seed < myoffset + mysize)
    {
        next_center.set_point(mypoints, global_seed-myoffset);
        next_center.id = global_seed;
        next_center.dist = std::numeric_limits<Real>::max();
        next_center.cell = 0;
    }
    else next_center.dist = 0;

    MPI_Allreduce(MPI_IN_PLACE, &next_center, 1, MPI_GLOBAL_POINT, MPI_ARGMAX, comm);
}

DistVoronoi::~DistVoronoi()
{
    MPI_Type_free(&MPI_GLOBAL_POINT);
    MPI_Op_free(&MPI_ARGMAX);
}

void DistVoronoi::add_next_center()
{
    int dim = mypoints.num_dimensions();

    Index cell = num_centers();
    centers.push_back(next_center.p);
    centerids.push_back(next_center.id);

    next_center.dist = 0;

    for (Index i = 0; i < mysize; ++i)
    {
        Real dist = mypoints.distance(i, next_center.p);

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

    next_center.set_point(mypoints, next_center.id-myoffset);

    MPI_Allreduce(MPI_IN_PLACE, &next_center, 1, MPI_GLOBAL_POINT, MPI_ARGMAX, comm);
}

void DistVoronoi::add_next_centers(Index count)
{
    int dim = mypoints.num_dimensions();
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

    CoverTree reptree(centers);
    reptree.build(1.3, 1);

    IndexPairVector ghostpairs;

    for (Index i = 0; i < mysize; ++i)
    {
        IndexVector ghostcells;
        reptree.radius_query(mypoints[i], dists[i] + 2*radius, ghostcells);

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

void DistVoronoi::load_alltoall_outbufs(const IndexVector& ids, const IndexVector& ptrs, const std::vector<int>& dests, std::vector<GlobalPoint>& sendbuf, std::vector<int>& sendcounts, std::vector<int>& sdispls) const
{
    Index m = num_centers();
    assert((ptrs.size() == m+1));
    assert((dests.size() == m));

    sendbuf.clear();
    sendcounts.resize(nprocs,0), sdispls.resize(nprocs);

    IndexVector rank_cell_map(m), rank_cell_counts(nprocs,0);

    Index totsend = 0;

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];
        sendcounts[dest] += (ptrs[i+1]-ptrs[i]);
        totsend += (ptrs[i+1]-ptrs[i]);

        rank_cell_map[i] = rank_cell_counts[dest];
        rank_cell_counts[dest]++;
    }

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0));
    sendbuf.resize(totsend);

    std::vector<int> sendptrs = sdispls;

    for (Index i = 0; i < m; ++i)
    {
        int dest = dests[i];

        for (Index ptr = ptrs[i]; ptr < ptrs[i+1]; ++ptr)
        {
            Index id = ids[ptr];
            Index loc = sendptrs[dest]++;

            sendbuf[loc].set_point(mypoints, id);
            sendbuf[loc].id = id+myoffset;
            sendbuf[loc].cell = rank_cell_map[i];
            sendbuf[loc].dist = dists[id];
        }
    }
}

std::string DistVoronoi::repr() const
{
    char buf[512];
    snprintf(buf, 512, "DistVoronoi(num_centers=%lld,next_center=%lld,separation=%.3f)", num_centers(), next_center_id(), center_separation());
    return std::string(buf);
}
