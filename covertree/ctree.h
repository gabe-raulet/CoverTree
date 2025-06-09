#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include <array>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include "metrics.h"

using Index = int64_t;
using IndexVector = std::vector<Index>;

template <class Distance, class Real, class Atom>
class CoverTree
{
    public:

        static inline constexpr Distance distance = Distance();

        using RealVector = std::vector<Real>;
        using AtomVector = std::vector<Atom>;

        struct Vertex
        {
            Index index, level;
            Real radius;
            IndexVector children, leaves;

            Vertex() {}
            Vertex(Index index, Real radius) : index(index), radius(radius) {}
        };

        using VertexVector = std::vector<Vertex>;

        CoverTree() {}
        CoverTree(Index n, Index d) : n(n), d(d) {}
        CoverTree(const CoverTree& rhs) = default;
        CoverTree& operator=(const CoverTree& rhs) = default;

        void build(const Atom *points, Real cover, Index leaf_size, int num_threads);
        Index radius_query(const Atom *points, const Atom *query, Real radius, IndexVector& neighbors, RealVector& dists) const;
        Index radius_neighbors_graph(const Atom *points, Real radius, std::vector<IndexVector>& neighbors, std::vector<RealVector>& dists, int num_threads) const;

        Index num_points() const { return n; }
        Index num_dimensions() const { return d; }
        Index num_vertices() const { return vertices.size(); }

        Index vertex_point(Index vertex) const { return vertices[vertex].index; }
        Index vertex_level(Index vertex) const { return vertices[vertex].level; }
        Real vertex_radius(Index vertex) const { return vertices[vertex].radius; }
        IndexVector vertex_children(Index vertex) const { return vertices[vertex].children; }
        IndexVector vertex_leaves(Index vertex) const { return vertices[vertex].leaves; }

    private:

        Index n, d;
        VertexVector vertices;
};

#include "ctree.hpp"

#endif
