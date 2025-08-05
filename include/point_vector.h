#ifndef POINT_VECTOR_H_
#define POINT_VECTOR_H_

#include "utils.h"
#include <assert.h>

class PointVector
{
    public:

        PointVector() : size(0), dim(1) {}
        PointVector(const AtomVector& atoms, int dim) : atoms(atoms), size(atoms.size()/dim), dim(dim) { assert((atoms.size() % dim == 0)); }
        PointVector(const Atom *atoms, Index size, int dim) : atoms(atoms, atoms + size*dim), size(size), dim(dim) {}
        PointVector(Index size, int dim) : atoms(size*dim), size(size), dim(dim) {}
        PointVector(const char *fname);

        Index num_atoms() const { return atoms.size(); }
        Index num_points() const { return size; }
        int num_dimensions() const { return dim; }

        Atom* data() { return atoms.data(); }
        const Atom* data() const { return atoms.data(); }
        const Atom* operator[](Index offset) const { return &atoms[offset*dim]; }

        Real distance(const Atom *p, const Atom *q) const;
        Real distance(Index p, const Atom *q) const { return distance((*this)[p], q); }
        Real distance(Index p, Index q) const { return distance((*this)[p], (*this)[q]); }

        PointIter begin(Index offset) const { return atoms.begin() + offset*dim; }
        PointIter end(Index offset) const { return begin(offset) + dim; }

        void resize(Index newsize) { atoms.resize(newsize*dim); size = newsize; }
        void resize(Index newsize, int newdim) { atoms.resize(newsize*newdim); size = newsize; dim = newdim; }

        void reserve(Index newcap) { atoms.reserve(newcap*dim); }
        void reserve(Index newcap, int newdim) { atoms.reserve(newcap*newdim); dim = newdim; }

        void clear() { atoms.clear(); size = 0; }
        void push_back(const Atom *pt) { atoms.insert(atoms.end(), pt, pt+dim); ++size; }

        void swap(PointVector& rhs) { std::swap(atoms, rhs.atoms); std::swap(size, rhs.size); std::swap(dim, rhs.dim); }


    protected:

        AtomVector atoms;
        Index size;
        int dim;
};

#endif
