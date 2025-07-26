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
    GlobalPoint(const Atom *pt, int dim, Index globidx, Index cell, Real dist);

    void set_point(const Atom *pt, int dim);
    void set_point(const PointVector& points, Index offset);

    Atom p[MAX_DIM];
    Index globidx;
    Index cell;
    Real dist;
};

#endif
