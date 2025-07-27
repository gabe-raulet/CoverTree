#ifndef POINT_VECTOR_H_
#define POINT_VECTOR_H_

#include "utils.h"

class PointVector
{
    public:

        PointVector() : size(0), dim(1) {}
        PointVector(int dim) : size(0), dim(dim) {}
        PointVector(const Atom *atoms, Index size, int dim) : atoms(atoms, atoms + size*dim), size(size), dim(dim) {}

        Index num_atoms() const { return atoms.size(); }
        Index num_points() const { return size; }
        int num_dimensions() const { return dim; }

        const Atom* data() const { return atoms.data(); }
        const Atom* operator[](Index offset) const { return &atoms[offset*dim]; }

        Real distance(const Atom *p, const Atom *q) const;
        Real distance(Index p, const Atom *q) const { return distance((*this)[p], q); }
        Real distance(Index p, Index q) const { return distance((*this)[p], (*this)[q]); }

        PointIter begin(Index offset) const { return atoms.begin() + offset*dim; }
        PointIter end(Index offset) const { return begin(offset) + dim; }

        void reserve(Index newcap) { atoms.reserve(newcap*dim); }
        void resize(Index newsize) { atoms.resize(newsize*dim); size = newsize; }
        void clear() { atoms.clear(); size = 0; }

        void push_back(const Atom *pt) { atoms.insert(atoms.end(), pt, pt+dim); ++size; }
        void set(Index offset, const Atom *pt) { std::copy(pt, pt+dim, &atoms[offset*dim]); }

        void read_fvecs(const char *fname);
        void write_fvecs(const char *fname) const;

        PointVector gather(const IndexVector& offsets) const;

        std::string repr() const;

    protected:

        AtomVector atoms;
        Index size;
        int dim;
};

#endif
