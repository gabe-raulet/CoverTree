#ifndef CTREE_H_
#define CTREE_H_

#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>
#include <mpi.h>
#include "bindutils.h"
#include "metricspace.h"
#include "covertree.h"

template <class Metric>
void bind_cover_tree(py::module_& m, const std::string& name)
{
    using Index = typename Metric::index_type;
    using Real = typename Metric::real_type;
    using Atom = typename Metric::atom_type;

    using IndexVector = typename Metric::IndexVector;
    using RealVector = typename Metric::RealVector;

    using covertree = CoverTree<Metric>;

    std::string vertex_name = name + "_Vertex";
    using Vertex = typename covertree::Vertex;

    py::class_<Vertex>(m, vertex_name.c_str())
        .def_readwrite("index", &Vertex::index)
        .def_readwrite("level", &Vertex::level)
        .def_readwrite("radius", &Vertex::radius)
        .def_readwrite("children", &Vertex::children)
        .def_readwrite("leaves", &Vertex::leaves)
        .def("__repr__", [](const Vertex& v)
                           {
                               std::ostringstream ss;
                               ss << "Vertex(index=" << v.index << ", level=" << v.level << ", radius=" << v.radius << ", children=[";
                               std::copy(v.children.begin(), v.children.end(), std::ostream_iterator<Index>(ss, ","));
                               ss << "], leaves=[";
                               std::copy(v.leaves.begin(), v.leaves.end(), std::ostream_iterator<Index>(ss, ","));
                               ss << "])";
                               return ss.str();
                           }
            );

    py::class_<covertree>(m, name.c_str())
        .def(py::init<const Metric&>())
        .def("num_points", &covertree::num_points)
        .def("num_dimensions", &covertree::num_dimensions)
        .def("radius_neighbors", [](const covertree& tree, NumpyArray<Atom>::type queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != tree.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const covertree& tree, NumpyArray<Index>::type_flexible queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != tree.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const covertree& tree, Real radius, int num_threads)
                                   {
                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors_dist", [](const covertree& tree, Real radius, py::object py_comm)
                                        {
                                            RealVector mydists; IndexVector myneighs, myptrs;
                                            MPI_Comm *comm;

                                            comm = PyMPIComm_Get(py_comm.ptr());

                                            if (!comm) throw py::error_already_set();

                                            tree.radius_neighbors(radius, mydists, myneighs, myptrs, *comm);

                                            return std::make_tuple(mydists, myneighs, myptrs);
                                        }
            )
        .def("get_packed", &covertree::get_packed)
        .def("build", &covertree::build, py::arg("cover") = 1.3, py::arg("leaf_size") = 40)
        .def("num_vertices", &covertree::num_vertices)
        .def("max_level", &covertree::max_level)
        .def("__getitem__", [](const covertree& tree, Index vertex) { return tree[vertex]; });

    using packed_covertree = PackedCoverTree<Metric>;
    std::string packed_name = std::string("Packed") + name;

    py::class_<packed_covertree>(m, packed_name.c_str())
        .def(py::init<const covertree&>())
        .def("num_points", &packed_covertree::num_points)
        .def("num_dimensions", &packed_covertree::num_dimensions)
        .def("radius_neighbors", [](const packed_covertree& tree, NumpyArray<Atom>::type queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != tree.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const packed_covertree& tree, NumpyArray<Index>::type_flexible queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != tree.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const packed_covertree& tree, Real radius, int num_threads)
                                   {
                                       RealVector dists; IndexVector neighs, ptrs;
                                       tree.radius_neighbors(radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors_dist", [](const packed_covertree& tree, Real radius, py::object py_comm)
                                        {
                                            RealVector mydists; IndexVector myneighs, myptrs;
                                            MPI_Comm *comm;

                                            comm = PyMPIComm_Get(py_comm.ptr());

                                            if (!comm) throw py::error_already_set();

                                            tree.radius_neighbors(radius, mydists, myneighs, myptrs, *comm);

                                            return std::make_tuple(mydists, myneighs, myptrs);
                                        }
            )
        .def("num_vertices", &packed_covertree::num_vertices)
        .def("max_level", &packed_covertree::max_level)
        .def("__getitem__", [](const packed_covertree& tree, Index vertex) { return tree[vertex]; });
}

template <class Atom>
void bind_cover_trees(py::module_& m, const std::string& atom_name)
{
    using Index = int64_t;
    using Real = float;

    using euclidean = Euclidean<Index, Real, Atom>;
    using manhattan = Manhattan<Index, Real, Atom>;
    using chebyshev = Chebyshev<Index, Real, Atom>;

    bind_cover_tree<euclidean>(m, std::string("CoverTreeEuclidean") + atom_name);
    bind_cover_tree<manhattan>(m, std::string("CoverTreeManhattan") + atom_name);
    bind_cover_tree<chebyshev>(m, std::string("CoverTreeChebyshev") + atom_name);
}

#endif
