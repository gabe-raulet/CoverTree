#ifndef METRICS_H_
#define METRICS_H_

#include <cmath>

template <class Atom>
concept is_real_atom = std::same_as<Atom, float> ||
                       std::same_as<Atom, double>;

template <class Atom>
concept is_int_atom = std::same_as<Atom, int8_t>  ||
                      std::same_as<Atom, uint8_t> ||
                      std::same_as<Atom, int32_t> ||
                      std::same_as<Atom, uint32_t>;

template <class Atom>
concept is_bool_atom = std::same_as<Atom, int8_t> ||
                       std::same_as<Atom, uint8_t>;

template <class Atom>
concept is_euclidean_atom = is_real_atom<Atom> || is_int_atom<Atom> || is_bool_atom<Atom>;

template <class Real, class Atom> requires is_real_atom<Real> && is_euclidean_atom<Atom>
struct EuclideanDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        Real val = 0;

        for (int i = 0; i < d; ++i)
        {
            Real delta = p[i] - q[i];
            val += delta * delta;
        }

        return std::sqrt(val);
    }
};

template <class Real, class Atom> requires is_real_atom<Real> && is_euclidean_atom<Atom>
struct ManhattanDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        Real val = 0;

        for (int i = 0; i < d; ++i)
        {
            val += std::abs(p[i] - q[i]);
        }

        return val;
    }
};

template <class Real, class Atom> requires is_real_atom<Real> && is_euclidean_atom<Atom>
struct AngularDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        Real pq = 0, pp = 0, qq = 0;

        for (int i = 0; i < d; ++i)
        {
            pq += p[i]*q[i];
            pp += p[i]*p[i];
            qq += q[i]*q[i];
        }


        if (pp*qq == 0)
            return 0;

        Real val = pq / std::sqrt(pp*qq);
        return acos(val) / M_PI;
    }
};

template <class Real, class Atom> requires is_real_atom<Real> && is_euclidean_atom<Atom>
struct ChebyshevDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        Real val = 0;

        for (int i = 0; i < d; ++i)
        {
            val = std::max(val, std::abs(p[i] - q[i]));
        }

        return val;
    }
};

template <class Real, class Atom> requires is_real_atom<Real> && is_bool_atom<Atom>
struct JaccardDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        int64_t both = 0;
        int64_t either = 0;

        for (int i = 0; i < d; ++i)
        {
            both += (int64_t)(p[i] == q[i] && p[i] != 0);
            either += (int64_t)(p[i] != q[i]);
        }

        if (either == 0) return 0;

        return  1.0 - ((both+0.0)/(either+0.0));
    }
};

template <class Real, class Atom> requires is_real_atom<Real>
struct HammingDistance
{
    Real operator()(const Atom *p, const Atom *q, int d) const
    {
        int64_t diffs = 0;

        for (int i = 0; i < d; ++i)
        {
            diffs += (int64_t)(p[i] != q[i]);
        }

        return (diffs+0.0)/d;
    }
};

#endif
