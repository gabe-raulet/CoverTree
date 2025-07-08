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
     *  - radius_query(const Atom*, Real, RealVector&, IndexVector&):
     *
     *      - Queries the point `query` against `metric`. The passed references `dists` and `neighs` should
     *        NOT be cleared by the implementation. The number of neighbors found is returned directly.
     *
     *  - radius_query(Index, Real, RealVector&, IndexVector&):
     *
     *      - Queries the local point in `metric` with index `query`.
     *
     *  - radius_neighbors(const Atom**, Index, Real, RealVector&, IndexVector&, IndexVector&, int):
     *
     *      - Queries multiple passed points against using `radius_query` and returns a CSR sparse matrix
     *        where (1) each row correspond to a query point (2) each column corresponds to a point in `metric`
     *        and (3) nonzeros are the distances to those points in `metric` that are within the radius
     *        threshold of a given query. The matrix is returned by reference with nonzero "values" (distances)
     *        stored in `dists`, column indices (neighbor indices) stored in `neighs`, and row pointers stored
     *        in `ptrs`. The algorithm supports OpenMP multithreading and the number of threads is passed
     *        as a parameter `num_threads`.
     *
     *      - Queries are passed as an array of `num_queries` pointers `queries` so that the other `radius_neighbors`
     *        variants can be implemented easily using this version as the main subroutine.
     *
     *  - radius_neighbors(const Atom*, Index, Real, RealVector&, IndexVector&, IndexVector&, int):
     *
     *      - Queries are stored contiguously one after the other starting from the pointer `queries`.
     *
     *  - radius_neighbors(const Index*, Index, Real, RealVector&, IndexVector&, IndexVector&, int):
     *
     *      - Queries the local points with indices stored in `queries`. The observant reader may notice
     *        that there is a potential problem if Atom and Index are the same type, as then the function above
     *        has the same type signature. This is currently avoided by statically asserting in the Metric class
     *        that this should not occur.
     *
     *  - radius_neighbors(Real, RealVector&, IndexVector&, IndexVector&, int):
     *
     *      - Queries all local points.
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
        Index radius_neighbors(const Index *queries, Index num_queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const;
        Index radius_neighbors(Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const { return radius_neighbors(metric.data(), num_points(), radius, dists, neighs, ptrs, num_threads); }
        /* Index radius_neighbors(Real radius, RealVector& mydists, IndexVector& myneighs, IndexVector& myptrs, MPI_Comm comm) const; */

        Index num_points() const { return metric.num_points(); }
        Index num_dimensions() const { return metric.num_dimensions(); }

        const Metric& get_metric() const { return metric; }

    protected:

        Metric metric;
};

#include "neighbors.hpp"

#endif
