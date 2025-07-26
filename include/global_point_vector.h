#ifndef GLOBAL_POINT_VECTOR_H_
#define GLOBAL_POINT_VECTOR_H_

#include "utils.h"
#include "point_vector.h"
#include "global_point.h"
#include <mpi.h>

class GlobalPointVector : public PointVector
{
    public:

        GlobalPointVector() = delete;
        GlobalPointVector(int dim);

        GlobalPoint operator[](Index offset) const;

        void reserve(Index newsize);
        void resize(Index newsize);
        void clear();

        void push_back(const GlobalPoint& pt);
        void set(Index offset, const GlobalPoint& pt);

        static void create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT);

    private:

        using PointVector::push_back;
        using PointVector::set;

        IndexVector cells;
        RealVector dists;
};

#endif
