#ifndef COVER_TREE_H_
#define COVER_TREE_H_

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
class CoverTree : public NearestNeighbors<Metric>
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

        void build(Real cover, Index leaf_size);
        virtual Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const final;

        struct Vertex
        {
            Index index, level;
            Real radius;
            IndexVector children, leaves;

            Vertex() {}
            Vertex(Index index, Real radius) : index(index), radius(radius) {}
        };

        using VertexVector = std::vector<Vertex>;

        Index num_vertices() const { return numverts; }
        Index max_level() const { return maxlevel; }

        Vertex operator[](Index vertex) const { return vertices[vertex]; }

    private:

        Index maxlevel, numverts;
        VertexVector vertices;
};

#include "covertree.hpp"

#endif
