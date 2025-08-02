#ifndef GLOBAL_POINT_H_
#define GLOBAL_POINT_H_

#include "point_vector.h"
#include <mpi.h>

#ifndef MAX_DIM
#error "MAX_DIM must be defined!"
#elif (MAX_DIM <= 0)
#error "MAX_DIM must be positive integer!"
#endif

struct GlobalPoint
{
    GlobalPoint() = default;
    GlobalPoint(const Atom *pt, int dim, Index id, Index cell, Real dist) : id(id), cell(cell), dist(dist) { set_point(pt, dim); }
    GlobalPoint& operator=(const GlobalPoint& rhs);

    void set_point(const Atom *pt, int dim) { std::copy(pt, pt+dim, p); }
    void set_point(const PointVector& points, Index offset) { set_point(points[offset], points.num_dimensions()); }

    std::string repr() const;

    Atom p[MAX_DIM];
    Index id;
    Index cell;
    Real dist;

    static void create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT, int dim);
};

using GlobalPointVector = std::vector<GlobalPoint>;

#endif
