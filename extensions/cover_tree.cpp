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

template <class Distance>
class cover_tree
{
    public:

        Index n, d;
        CoverTree<Distance> tree;

        cover_tree(py::array_t<Real> points_py)
        {
            py::buffer_info info = points_py.request();
            n = info.shape[0], d = info.shape[1];
        }

        void build_index(py::array_t<Real> points_py, Real cover, int leaf_size, int num_threads)
        {
            tree.build(npmem(points_py), n, d, cover, leaf_size);
        }

        std::tuple<RealVector, IndexVector>
        range_query(py::array_t<Real> points_py, py::array_t<Real> query_py, Real radius) const
        {
            IndexVector neighbors;
            RealVector dists;

            tree.range_query(npmem(points_py), npmem(query_py), radius, neighbors, dists);

            return std::make_tuple(dists, neighbors);
        }

        std::tuple<RealVector, IndexVector, IndexVector>
        radius_neighbors_graph(py::array_t<Real> points_py, Real radius, int num_threads)
        {
            RealVector dists;
            IndexVector rowptrs(n+1), colids;

            std::vector<IndexVector> neighbors_graph(n);
            std::vector<RealVector> dists_graph(n);

            Real *points = npmem(points_py);

            omp_set_num_threads(num_threads);

            Index nz = 0;

            #pragma omp parallel for reduction(+:nz)
            for (Index i = 0; i < n; ++i)
            {
                nz += tree.range_query(points, &points[i*d], radius, neighbors_graph[i], dists_graph[i]);
            }

            colids.reserve(nz);
            dists.reserve(nz);

            rowptrs[0] = 0;

            for (Index i = 0; i < n; ++i)
            {
                std::copy(neighbors_graph[i].begin(), neighbors_graph[i].end(), std::back_inserter(colids));
                std::copy(dists_graph[i].begin(), dists_graph[i].end(), std::back_inserter(dists));
                rowptrs[i+1] = colids.size();
            }

            return std::make_tuple(dists, colids, rowptrs);
        }
};

PYBIND11_MODULE(cover_tree, m)
{
    py::class_<cover_tree<EuclideanDistance>>(m, "cover_tree_euclidean")
        .def(py::init<py::array_t<Real>>())
        .def("build_index", &cover_tree<EuclideanDistance>::build_index)
        .def("range_query", &cover_tree<EuclideanDistance>::range_query)
        .def("radius_neighbors_graph", &cover_tree<EuclideanDistance>::radius_neighbors_graph);

    py::class_<cover_tree<ManhattanDistance>>(m, "cover_tree_manhattan")
        .def(py::init<py::array_t<Real>>())
        .def("build_index", &cover_tree<ManhattanDistance>::build_index)
        .def("range_query", &cover_tree<ManhattanDistance>::range_query)
        .def("radius_neighbors_graph", &cover_tree<ManhattanDistance>::radius_neighbors_graph);

    py::class_<cover_tree<ChebyshevDistance>>(m, "cover_tree_chebyshev")
        .def(py::init<py::array_t<Real>>())
        .def("build_index", &cover_tree<ChebyshevDistance>::build_index)
        .def("range_query", &cover_tree<ChebyshevDistance>::range_query)
        .def("radius_neighbors_graph", &cover_tree<ChebyshevDistance>::radius_neighbors_graph);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
