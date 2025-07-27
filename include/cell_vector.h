#ifndef CELL_VECTOR_H_
#define CELL_VECTOR_H_

#include "point_vector.h"

struct GlobalPoint;

class CellVector : public PointVector
{
    public:

        CellVector() : PointVector() {}
        CellVector(int dim) : PointVector(dim) {}

        void reserve(Index newcap);
        void resize(Index newsize);
        void clear();

        void push_back(const GlobalPoint& p);
        void sort_by_dists();

        virtual Index index(Index offset) const override { return indices[offset]; }

    private:

        IndexVector indices;
        RealVector dists;
};

#endif
