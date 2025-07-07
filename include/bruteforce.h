#ifndef BRUTE_FORCE_H_
#define BRUTE_FORCE_H_

#include <stdio.h>
#include <string>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <deque>
#include <limits>
#include <cmath>
#include <assert.h>
#include <omp.h>
#include <mpi.h>
#include "metricspace/metricspace.h"

template <class Metric>
class BruteForce
{
    public:

        using Index = typename Metric::index_type;
        using Real = typename Metric::real_type;
        using Atom = typename Metric::atom_type;
        using Triple = typename Metric::Triple;

        using IndexVector = typename Metric::IndexVector;
        using RealVector = typename Metric::RealVector;
        using AtomVector = typename Metric::AtomVector;
        using TripleVector = typename Metric::TripleVector;

        BruteForce() {}
        BruteForce(const Metric& metric) : metric(metric) {}

        Index radius_query(const Atom *query, Real radius, IndexVector& neighs, RealVector& dists) const;
        Index radius_query(Index query, Real radius, IndexVector& neighs, RealVector& dists) const { return radius_query(metric[query], radius, neighs, dists); }

        Index num_points() const { return metric.num_points(); }
        Index num_dimensions() const { return metric.num_dimensions(); }

        const Metric& get_metric() const { return metric; }

    private:

        Metric metric;
};

template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_query(const Atom *query, Real radius, IndexVector& neighs, RealVector& dists) const
{
    Index found = 0;
    Index n = metric.num_points();

    for (Index i = 0; i < n; ++i)
    {
        Real dist = metric.distance(i, query);

        if (dist <= radius)
        {
            neighs.push_back(i);
            dists.push_back(dist);
            found++;
        }
    }

    return found;
}

#endif
