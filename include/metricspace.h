#ifndef METRIC_SPACE_H_
#define METRIC_SPACE_H_

#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <type_traits>

template <class Index, class Real, class Atom>
class MetricSpace
{
    /*
     * MetricSpace is an abstract base class for metric spaces whose points are
     * fixed-length vectors. MetricSpace is templated with an integer type `Index`
     * for indexing points (and integer values in general), a numeric type `Real`
     * for representing distances between points, and a type `Atom` which is the
     * underyling point vector type.
     *
     * Points are assumed to be fixed-length `Atom` vectors with dimension `dim`.
     * For example, 3-dimensional euclidean space could be represented with the
     * template MetricSpace<int64_t, double, float>, and a `dim` variable equal
     * to 3, where distances are represented with 64-bit floating point values
     * and points are represented by 3 32-bit floating point values.
     *
     * MetricSpace has two pure virtual functions: "metric(void)" and "distance(const Atom*, const Atom*)".
     * "metric(void)" is simply overriden with the name of the metric used, and "distance(..)" is overriden
     * with an implementation of the distance metric function. All other functions are non-virtual.
     */

    public:

        static_assert(!std::is_same<Index, Atom>::value, "Index cannot equal Atom at this time, as it complicates some function implementations and should be rare anyways.");

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

        virtual constexpr const char* metric() const = 0;
        virtual Real distance(const Atom *p, const Atom *q) const = 0;

        Real distance(Index p, const Atom *q) const { return distance(point(p), q); }
        Real distance(Index p, Index q) const { return distance(point(p), point(q)); }

        Index num_points() const { return size; }
        Index num_dimensions() const { return dim; }
        Index num_atoms() const { return size*dim; }

        Atom* data() { return atoms.data(); }
        const Atom* data() const { return atoms.data(); }

        void* mem() { return static_cast<void*>(atoms.data()); }
        const void* mem() const { return static_cast<const void*>(atoms.data()); }

    protected:

        AtomVector atoms;
        Index size, dim;
};

#include "metricspace.hpp"

#endif
