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
#include "neighbors.h"

template <class Metric>
class BruteForce : public NearestNeighbors<Metric>
{
    public:

        using Base = NearestNeighbors<Metric>;
        using Base::Base;
        using Base::radius_query;
        using Base::radius_neighbors;

        using Index = typename Metric::index_type;
        using Real = typename Metric::real_type;
        using Atom = typename Metric::atom_type;
        using Triple = typename Metric::Triple;

        using IndexVector = typename Metric::IndexVector;
        using RealVector = typename Metric::RealVector;
        using AtomVector = typename Metric::AtomVector;
        using TripleVector = typename Metric::TripleVector;

        virtual Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const final
        {
            Index found = 0;
            Index n = Base::metric.num_points();

            for (Index i = 0; i < n; ++i)
            {
                Real dist = Base::metric.distance(i, query);

                if (dist <= radius)
                {
                    neighs.push_back(i);
                    dists.push_back(dist);
                    found++;
                }
            }

            return found;
        }

};

#endif
