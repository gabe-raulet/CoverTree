#include "global_point.h"
#include <assert.h>

void GlobalPoint::create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT, int dim)
{
    assert((dim <= MAX_DIM));

    int blklens[4] = {dim,1,1,1};
    MPI_Aint disps[4] = {offsetof(GlobalPoint,p), offsetof(GlobalPoint,id), offsetof(GlobalPoint,cell), offsetof(GlobalPoint,dist)};
    MPI_Datatype types[4] = {MPI_ATOM, MPI_INDEX, MPI_INDEX, MPI_REAL};
    MPI_Type_create_struct(4, blklens, disps, types, MPI_GLOBAL_POINT);
    MPI_Type_commit(MPI_GLOBAL_POINT);
}

GlobalPoint& GlobalPoint::operator=(const GlobalPoint& rhs)
{
    id = rhs.id;
    cell = rhs.cell;
    dist = rhs.dist;
    std::copy(rhs.p, rhs.p+MAX_DIM, p);
    return *this;
}

std::string GlobalPoint::repr() const
{
    char buf[512];
    snprintf(buf, 512, "GlobalPoint(id=%lld,cell=%lld,dist=%.3f)", id, cell, dist);
    return std::string(buf);
}
