#ifndef POINT_VECTOR_H_
#define POINT_VECTOR_H_

#include "utils.h"

class PointVector
{
    public:

        PointVector() = delete;
        PointVector(int dim);
        PointVector(const Atom *atoms, Index size, int dim);

        Index num_atoms() const;
        Index num_points() const;
        int num_dimensions() const;

        const Atom* data() const;
        const Atom* operator[](Index offset) const;

        Real distance(const Atom *p, const Atom *q) const;
        Real distance(Index p, const Atom *q) const;
        Real distance(Index p, Index q) const;

        PointIter begin(Index offset) const;
        PointIter end(Index offset) const;

        void reserve(Index newsize);
        void resize(Index newsize);
        void clear();

        void push_back(const Atom *pt);
        void set(Index offset, const Atom *pt);

        void read_fvecs(const char *fname);
        std::string repr() const;

    protected:

        AtomVector atoms;
        Index size;
        int dim;
};

#endif
