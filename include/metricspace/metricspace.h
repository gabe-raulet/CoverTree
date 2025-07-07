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

template <class Index, class Real, class Atom>
class Euclidean : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "euclidean"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Real val = 0;
            Index d = this->num_dimensions();

            for (Index i = 0; i < d; ++i)
            {
                Real delta = p[i] - q[i];
                val += delta * delta;
            }

            return std::sqrt(val);
        }
};

template <class Index, class Real, class Atom>
class Levenshtein : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "levenshtein"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Index d = this->num_dimensions();
            typename MetricSpace<Index, Real, Atom>::IndexVector v0(d+1), v1(d+1);

            for (Index i = 0; i <= d; ++i)
                v0[i] = i;

            for (Index i = 0; i < d; ++i)
            {
                v1[0] = i+1;

                for (Index j = 0; j < d; ++j)
                {
                    Index del_cost = v0[j+1] + 1;
                    Index ins_cost = v1[j] + 1;
                    Index sub_cost = (p[i] == q[i])? v0[j] : v0[j] + 1;

                    v1[j+1] = std::min(del_cost, std::min(ins_cost, sub_cost));
                }

                std::swap(v0, v1);
            }

            return v0[d];
        }
};

#endif
