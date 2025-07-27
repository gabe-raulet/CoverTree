#ifndef GLOBAL_POINT_VECTOR_H_
#define GLOBAL_POINT_VECTOR_H_

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
    GlobalPoint(const Atom *pt, int dim, Index id, Index cell, Real dist) : id(id), cell(cell), dist(dist) { set_point(pt, dim); }

    void set_point(const Atom *pt, int dim) { std::copy(pt, pt+dim, p); }
    void set_point(const PointVector& points, Index offset) { set_point(points[offset], points.num_dimensions()); }

    Atom p[MAX_DIM];
    Index id;
    Index cell;
    Real dist;

    static void create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT, int dim);
};

class GlobalPointVector : public PointVector
{
    public:

        GlobalPointVector() : PointVector(1) {}
        GlobalPointVector(int dim) : PointVector(dim) {}
        GlobalPointVector(const PointVector& mypoints, Index cell_init, Real dist_init, MPI_Comm comm);

        GlobalPoint operator[](Index offset) const;

        Index getid(Index offset) const { return ids[offset]; }
        Index getcell(Index offset) const { return cells[offset]; }
        Real getdist(Index offset) const { return dists[offset]; }

        Index& getid(Index offset) { return ids[offset]; }
        Index& getcell(Index offset) { return cells[offset]; }
        Real& getdist(Index offset) { return dists[offset]; }

        void reserve(Index newcap);
        void resize(Index newsize);
        void clear();

        void push_back(const GlobalPoint& pt);
        void set(Index offset, const GlobalPoint& pt);

        void read_fvecs(const char *fname, MPI_Comm comm);

        std::string repr() const;

    private:

        using PointVector::push_back;
        using PointVector::set;

        IndexVector ids;
        IndexVector cells;
        RealVector dists;
};

#endif
