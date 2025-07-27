#include "global_point_vector.h"
#include <assert.h>
#include <numeric>

GlobalPointVector::GlobalPointVector(const PointVector& mypoints, Index cell_init, Real dist_init, MPI_Comm comm)
    : PointVector(mypoints),
      ids(mypoints.num_points()),
      cells(mypoints.num_points(), cell_init),
      dists(mypoints.num_points(), dist_init)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);

    Index mysize = mypoints.num_points();
    Index myoffset;

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    std::iota(ids.begin(), ids.end(), myoffset);
}

void GlobalPointVector::create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT) const
{
    assert((dim <= MAX_DIM));

    int blklens[4] = {dim,1,1,1};
    MPI_Aint disps[4] = {offsetof(GlobalPoint,p), offsetof(GlobalPoint,id), offsetof(GlobalPoint,cell), offsetof(GlobalPoint,dist)};
    MPI_Datatype types[4] = {MPI_ATOM, MPI_INDEX, MPI_INDEX, MPI_REAL};
    MPI_Type_create_struct(4, blklens, disps, types, MPI_GLOBAL_POINT);
    MPI_Type_commit(MPI_GLOBAL_POINT);
}

GlobalPoint GlobalPointVector::operator[](Index offset) const
{
    return GlobalPoint(PointVector::operator[](offset), num_dimensions(), ids[offset], cells[offset], dists[offset]);
}

void GlobalPointVector::reserve(Index newcap)
{
    PointVector::reserve(newcap);
    ids.reserve(newcap);
    cells.reserve(newcap);
    dists.reserve(newcap);
}

void GlobalPointVector::resize(Index newsize)
{
    PointVector::resize(newsize);
    ids.resize(newsize);
    cells.resize(newsize);
    dists.resize(newsize);
}

void GlobalPointVector::clear()
{
    PointVector::clear();
    ids.clear();
    cells.clear();
    dists.clear();
}

void GlobalPointVector::push_back(const GlobalPoint& pt)
{
    PointVector::push_back(pt.p);
    ids.push_back(pt.id);
    cells.push_back(pt.cell);
    dists.push_back(pt.dist);
}

void GlobalPointVector::set(Index offset, const GlobalPoint& pt)
{
    PointVector::set(offset, pt.p);
    ids[offset] = pt.id;
    cells[offset] = pt.cell;
    dists[offset] = pt.dist;
}
