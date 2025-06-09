#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include <array>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include "metrics.h"

using Real = float;
using Index = int64_t;

using RealVector = std::vector<Real>;
using IndexVector = std::vector<Index>;

template <class Distance>
class CoverTree
{
    public:

        static inline constexpr Distance distance = Distance();

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
        CoverTree(const CoverTree& rhs) = default;
        CoverTree& operator=(const CoverTree& rhs) = default;

        void swap(CoverTree& rhs) { std::swap(vertices, rhs.vertices); std::swap(d, rhs.d); std::swap(n, rhs.n); }

        void build(const Real *points, Index n, int d, Real cover, int leaf_size);

        Index range_query(const Real *points, const Real *query, Real radius, IndexVector& neighbors, RealVector& dists) const;

        Index num_points() const { return n; }
        Index num_vertices() const { return vertices.size(); }
        Index vertex_point(Index vertex) const { return vertices[vertex].index; }
        Index vertex_level(Index vertex) const { return vertices[vertex].level; }
        Real vertex_radius(Index vertex) const { return vertices[vertex].radius; }
        IndexVector vertex_children(Index vertex) const { return vertices[vertex].children; }
        IndexVector vertex_leaves(Index vertex) const { return vertices[vertex].leaves; }

    private:

        int d;
        Index n;
        VertexVector vertices;
};

#include "ctree.hpp"

#endif
