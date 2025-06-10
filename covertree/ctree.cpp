//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <sstream>
#include <assert.h>
#include <omp.h>
#include "utils.h"
#include "metrics.h"
#include "ctree.h"

namespace py = pybind11;

template <class Distance, class Real, class Atom>
class CoverTreeWrapper
{
    public:

        using RealVector = std::vector<Real>;

        CoverTreeWrapper(py::array_t<Atom> points_py)
            : tree(npsize(points_py, 0), npsize(points_py, 1)),
              points(npmem(points_py)) {}

        void build_index(Real cover, Index leaf_size, int num_threads)
        {
            tree.build(points, cover, leaf_size, num_threads);
        }

        std::tuple<RealVector, IndexVector>
        radius_query(const Atom *query, Real radius) const
        {
            IndexVector neighbors;
            RealVector dists;

            tree.radius_query(points, query, radius, neighbors, dists);

            return std::make_tuple(dists, neighbors);
        }

        std::tuple<RealVector, IndexVector>
        knn_query(const Atom *query, Index k) const
        {
            IndexVector neighbors;
            RealVector dists;

            tree.knn_query(points, query, k, neighbors, dists);

            return std::make_tuple(dists, neighbors);
        }

        std::tuple<RealVector, IndexVector, IndexVector>
        radius_neighbors_graph(Real radius, int num_threads) const
        {
            std::vector<IndexVector> neighbors;
            std::vector<RealVector> dists;

            Index nz = tree.radius_neighbors_graph(points, radius, neighbors, dists, num_threads);

            Index n = tree.num_points();

            RealVector data(nz);
            IndexVector rowptrs(n+1), colids(nz);

            Index p = 0;
            auto data_itr = data.begin();
            auto colids_itr = colids.begin();

            for (Index i = 0; i < n; ++i)
            {
                rowptrs[i] = p;
                Index count = neighbors[i].size();

                std::copy(neighbors[i].begin(), neighbors[i].end(), colids_itr);
                std::copy(dists[i].begin(), dists[i].end(), data_itr);

                p += count;
                colids_itr += count;
                data_itr += count;
            }

            rowptrs[n] = nz;

            return std::make_tuple(data, colids, rowptrs);
        }

        CoverTree<Distance, Real, Atom> tree;
        const Atom *points;
};

template <class Distance, class Real, class Atom>
void bind_cover_tree_wrapper(py::module_& m, const std::string& name)
{
    using Tree = CoverTreeWrapper<Distance, Real, Atom>;

    py::class_<Tree>(m, name.c_str())
        .def(py::init<py::array_t<Atom>>())
        .def("build_index", &Tree::build_index)
        .def("radius_query", [](const Tree& tree, py::array_t<Atom> query, Real radius) { return tree.radius_query(npmem(query), radius); })
        .def("radius_neighbors_graph", &Tree::radius_neighbors_graph)
        .def("knn_query", [](const Tree& tree, py::array_t<Atom> query, int k) { return tree.knn_query(npmem(query), k); })
        .def("vertex_point", [](const Tree& tree, Index vertex) { return tree.tree.vertex_point(vertex); })
        .def("vertex_level", [](const Tree& tree, Index vertex) { return tree.tree.vertex_level(vertex); })
        .def("vertex_radius", [](const Tree& tree, Index vertex) { return tree.tree.vertex_radius(vertex); })
        .def("vertex_children", [](const Tree& tree, Index vertex) { return tree.tree.vertex_children(vertex); })
        .def("vertex_leaves", [](const Tree& tree, Index vertex) { return tree.tree.vertex_leaves(vertex); });
}


PYBIND11_MODULE(ctree, m)
{
    bind_cover_tree_wrapper<ManhattanDistance<float, float>, float, float>(m, "cover_tree_l1");
    bind_cover_tree_wrapper<EuclideanDistance<float, float>, float, float>(m, "cover_tree_l2");
    bind_cover_tree_wrapper<ChebyshevDistance<float, float>, float, float>(m, "cover_tree_linf");
    bind_cover_tree_wrapper<AngularDistance<float, float>, float, float>(m, "cover_tree_angular");
    bind_cover_tree_wrapper<JaccardDistance<float, uint8_t>, float, uint8_t>(m, "cover_tree_jaccard");
    bind_cover_tree_wrapper<HammingDistance<float, uint8_t>, float, uint8_t>(m, "cover_tree_hamming");
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
