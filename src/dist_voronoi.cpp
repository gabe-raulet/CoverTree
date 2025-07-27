#include "dist_voronoi.h"

void DistVoronoi::mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype)
{
    GlobalPoint *in = (GlobalPoint *)_in;
    GlobalPoint *inout = (GlobalPoint *)_inout;

    for (int i = 0; i < *len; ++i)
        if (in[i].dist > inout[i].dist)
            inout[i] = in[i];
}

DistVoronoi::DistVoronoi(const PointVector& points, Index global_seed, MPI_Comm comm)
    : mypoints(points, global_seed, std::numeric_limits<Real>::max(), comm),
      centers(points.num_dimensions()),
      comm(comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    mypoints.create_mpi_type(&MPI_GLOBAL_POINT);
    MPI_Op_create(&mpi_argmax, 0, &MPI_ARGMAX);

    Index mysize = mypoints.num_points();
    Index myoffset = mypoints.getid(0);

    if (myoffset <= global_seed && global_seed < myoffset + mysize)
        next_center = mypoints[global_seed-myoffset];
    else
        next_center.dist = 0;

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
    centers.push_back(next_center);

    next_center.dist = 0;
    Index mysize = mypoints.num_points();
    Index myoffset = mypoints.getid(0);

    for (Index i = 0; i < mysize; ++i)
    {
        Real dist = mypoints.distance(i, next_center.p);

        if (dist < mypoints.getdist(i))
        {
            mypoints.getdist(i) = dist;
            mypoints.getcell(i) = cell;
        }

        if (mypoints.getdist(i) > next_center.dist)
        {
            next_center.id = mypoints.getid(i);
            next_center.dist = mypoints.getdist(i);
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

std::string DistVoronoi::repr() const
{
    char buf[512];
    snprintf(buf, 512, "DistVoronoi(num_centers=%lld,next_center=%lld,separation=%.3f)", num_centers(), next_center_id(), center_separation());
    return std::string(buf);
}
