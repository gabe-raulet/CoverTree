#ifndef NEIGHBORS_H_
#define NEIGHBORS_H_

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
#include "utils.h"
#include "metricspace.h"

template <class Metric>
class NearestNeighbors
{
    /*
     * NearestNeighbors is an abstract base class implementing the interface and helper
     * functions of a nearest neighbor indexing algorithm. The class takes a concrete class
     * derived from a metric space `Metric` (a class template) in the constructor and
     * implements various nearest neighbor query routines.
     *
     * The main radius query routine `radius_query(const Atom* query, Real, RealVector&, IndexVector&)`
     * is pure virtual and must be implemented in a derived class with a query algorithm.
     * `radius_query` takes a point `query` and threshold `radius` as inputs and finds all the points
     * in `metric` that are within a distance `radius` of `query` using the distance method implemented
     * by the `Metric` type. The indices are returned by reference through `neighs` and the corresponding
     * distances through `dists`.
     *
     * A number of other non-virtual routines are implemented that use the pure virtual `radius_query` implementation:
     *
     *  - radius_query(Index, Real, RealVector&, IndexVector&):
     *
     *     - Queries the local point in `metric` with index `query`
     *
     *  - radius_neighbors(const Atom**, Index, Real, RealVector&, IndexVector&, IndexVector&,
     */

    public:

        using Index = typename Metric::index_type;
        using Real = typename Metric::real_type;
        using Atom = typename Metric::atom_type;
        using Triple = typename Metric::Triple;

        using IndexVector = typename Metric::IndexVector;
        using RealVector = typename Metric::RealVector;
        using AtomVector = typename Metric::AtomVector;
        using TripleVector = typename Metric::TripleVector;

        NearestNeighbors(const Metric& metric) : metric(metric) {}

        virtual ~NearestNeighbors() {}

        virtual Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const = 0;
        Index radius_query(Index query, Real radius, RealVector& dists, IndexVector& neighs) const { return radius_query(metric[query], radius, dists, neighs); }

        Index radius_neighbors(const Atom **queries, Index num_queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const;
        Index radius_neighbors(const Atom *queries, Index num_queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const;
        Index radius_neighbors(const IndexVector& queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const;
        Index radius_neighbors(Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const { return radius_neighbors(metric.data(), num_points(), radius, dists, neighs, ptrs, num_threads); }
        Index radius_neighbors(Real radius, RealVector& mydists, IndexVector& myneighs, IndexVector& myptrs, MPI_Comm comm) const;

        Index num_points() const { return metric.num_points(); }
        Index num_dimensions() const { return metric.num_dimensions(); }

        const Metric& get_metric() const { return metric; }

    protected:

        Metric metric;
};

#include "neighbors.hpp"

#endif
