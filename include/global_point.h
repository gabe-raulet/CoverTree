#ifndef GLOBAL_POINT_H_
#define GLOBAL_POINT_H_

#include "utils.h"
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
    GlobalPoint(const Atom *pt, int dim, Index globidx, Index cell, Real dist) : globidx(globidx), cell(cell), dist(dist) { set_point(pt, dim); }

    void set_point(const Atom *pt, int dim) { std::copy(pt, pt+dim, p); }
    void set_point(const PointVector& points, Index offset) { set_point(points[offset], points.num_dimensions()); }

    Atom p[MAX_DIM];
    Index globidx;
    Index cell;
    Real dist;
};

#endif
