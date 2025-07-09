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
        virtual Real get_radius(Index vertex) const = 0;
        virtual Index get_index(Index vertex) const = 0;
        virtual Index get_level(Index vertex) const = 0;
        virtual const Index* get_children(Index vertex, Index& nchild) const = 0;
        virtual const Index* get_leaves(Index vertex, Index& nleaf) const = 0;

        Index radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const;

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
class PackedCoverTree;

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

        using Vertex = typename Base::Vertex;
        using VertexVector = typename Base::VertexVector;

        virtual Vertex operator[](Index vertex) const final { return vertices[vertex]; }

        virtual Real get_radius(Index vertex) const final { return vertices[vertex].radius; }
        virtual Index get_index(Index vertex) const final { return vertices[vertex].index; }
        virtual Index get_level(Index vertex) const final { return maxlevel - vertices[vertex].level; }
        virtual const Index* get_children(Index vertex, Index& nchild) const final { nchild = vertices[vertex].children.size(); return vertices[vertex].children.data(); }
        virtual const Index* get_leaves(Index vertex, Index& nleaf) const final { nleaf = vertices[vertex].leaves.size(); return vertices[vertex].leaves.data(); }

        virtual void build(Real cover, Index leaf_size) final;

        PackedCoverTree<Metric> get_packed() const { return PackedCoverTree<Metric>(*this); }

    private:

        VertexVector vertices;
};

template <class Metric>
class PackedCoverTree: public CoverTreeInterface<Metric>
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

        using BaseVertex = typename Base::Vertex;

        struct Vertex
        {
            Index index, level, childptr, leafptr, nchild, nleaf;
            Real radius;
        };

        using VertexVector = std::vector<Vertex>;

        virtual Real get_radius(Index vertex) const final { return vertices[vertex].radius; }
        virtual Index get_index(Index vertex) const final { return vertices[vertex].index; }
        virtual Index get_level(Index vertex) const final { return maxlevel - vertices[vertex].level; }

        virtual const Index* get_children(Index vertex, Index& nchild) const final
        {
            nchild = vertices[vertex].nchild;
            return &ids[vertices[vertex].childptr];
        }

        virtual const Index* get_leaves(Index vertex, Index& nleaf) const final
        {
            nleaf = vertices[vertex].nleaf;
            return &ids[vertices[vertex].leafptr];
        }

        virtual BaseVertex operator[](Index vertex) const final
        {
            Index nchild, nleaf;
            Index index = get_index(vertex);
            Index level = get_level(vertex);
            Real radius = get_radius(vertex);
            const Index *children = get_children(vertex, nchild);
            const Index *leaves = get_leaves(vertex, nleaf);
            return BaseVertex(index, level, radius, IndexVector(children, children+nchild), IndexVector(leaves, leaves+nleaf));
        }

        virtual void build(Real cover, Index leaf_size) final { throw std::runtime_error("Not supported!"); }

        PackedCoverTree(const CoverTree<Metric>& tree);

    private:

        VertexVector vertices;
        IndexVector ids;
};

#include "covertree.hpp"

#endif
