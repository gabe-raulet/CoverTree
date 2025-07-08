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
class CoverTreeInterface : public NearestNeighbors<Metric>
{
    public:

        using Base = NearestNeighbors<Metric>;
        using Base::Base;
        using Base::radius_query;
        using Base::radius_neighbors;
        using Base::metric;

        using Index = typename Metric::index_type;
        using Real = typename Metric::real_type;
        using Atom = typename Metric::atom_type;
        using Triple = typename Metric::Triple;

        using IndexVector = typename Metric::IndexVector;
        using RealVector = typename Metric::RealVector;
        using AtomVector = typename Metric::AtomVector;
        using TripleVector = typename Metric::TripleVector;

        virtual void build(Real cover, Index leaf_size) = 0;
        virtual Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const = 0;

        struct Vertex
        {
            Index index, level;
            Real radius;
            IndexVector children, leaves;

            Vertex() {}
            Vertex(Index index, Real radius) : index(index), radius(radius) {}
            Vertex(Index index, Index level, Real radius, IndexVector children, IndexVector leaves) : index(index), level(level), radius(radius), children(children), leaves(leaves) {}
        };

        using VertexVector = std::vector<Vertex>;

        Index num_vertices() const { return numverts; }
        Index max_level() const { return maxlevel; }

        virtual Vertex operator[](Index vertex) const = 0;

    protected:

        Index maxlevel, numverts;
};


template <class Metric>
class CoverTree: public CoverTreeInterface<Metric>
{
    public:

        using Base = CoverTreeInterface<Metric>;
        using Base::Base;
        using Base::radius_query;
        using Base::radius_neighbors;
        using Base::num_points;
        using Base::maxlevel;
        using Base::numverts;
        using Base::metric;

        using Index = typename Metric::index_type;
        using Real = typename Metric::real_type;
        using Atom = typename Metric::atom_type;
        using Triple = typename Metric::Triple;

        using IndexVector = typename Metric::IndexVector;
        using RealVector = typename Metric::RealVector;
        using AtomVector = typename Metric::AtomVector;
        using TripleVector = typename Metric::TripleVector;

        virtual void build(Real cover, Index leaf_size) final;
        virtual Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const final;

        using Vertex = typename Base::Vertex;
        using VertexVector = typename Base::VertexVector;

        virtual Vertex operator[](Index vertex) const { return vertices[vertex]; }

    private:

        VertexVector vertices;
};

#include "covertree.hpp"

#endif
