#ifndef METRIC_SPACE_H_
#define METRIC_SPACE_H_

#include <cmath>
#include <vector>
#include <tuple>
#include <assert.h>
#include <omp.h>
#include <mpi.h>

template <class Index, class Real, class Atom>
class MetricSpace
{
    public:

        virtual constexpr const char* metric() const = 0;

        using index_type = Index;
        using real_type = Real;
        using atom_type = Atom;

        using IndexVector = std::vector<Index>;
        using RealVector = std::vector<Real>;
        using AtomVector = std::vector<Atom>;

        using Triple = std::tuple<Index, Index, Real>;
        using TripleVector = std::vector<Triple>;

        using iterator = AtomVector::iterator;
        using const_iterator = AtomVector::const_iterator;

        MetricSpace(const Atom *data, Index size, Index dim) : atoms(data, data + size*dim), size(size), dim(dim) {}
        MetricSpace(const AtomVector& atoms, Index size, Index dim) : MetricSpace(atoms.data(), size, dim) {}
        MetricSpace(const MetricSpace& rhs) = default;
        MetricSpace& operator=(const MetricSpace& rhs) = default;

        virtual ~MetricSpace() {}

        const Atom* point(Index idx) const { return &atoms[idx*dim]; }
        const Atom* operator[](Index idx) const { return &atoms[idx*dim]; }

        iterator begin(Index idx = 0) { return atoms.begin() + idx*dim; }
        iterator end(Index idx = 0) { return begin(idx) + dim; }

        const_iterator cbegin(Index idx = 0) const { return atoms.begin() + idx*dim; }
        const_iterator cend(Index idx = 0) const { return begin(idx) + dim; }

        virtual Real distance(const Atom *p, const Atom *q) const = 0;
        virtual Real distance(Index p, const Atom *q) const { return distance(point(p), q); }
        virtual Real distance(Index p, Index q) const { return distance(point(p), point(q)); }

        Index num_points() const { return size; }
        Index num_dimensions() const { return dim; }
        Index num_atoms() const { return size*dim; }

        void* data() {return static_cast<void*>(atoms.data()); }

    protected:

        AtomVector atoms;
        Index size, dim;
};

#include "metricspace.hpp"

#endif
