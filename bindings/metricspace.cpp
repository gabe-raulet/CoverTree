#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <mpi.h>
#include "utils.h"
#include "dist_point_vector.h"

namespace py = pybind11;

template <class T>
struct NumpyArray
{
    using type = py::array_t<T, py::array::c_style>;
    using type_flexible = py::array_t<T, py::array::c_style | py::array::forcecast>;
};

MPI_Comm* get_py_comm(py::object py_comm)
{
     MPI_Comm *comm = PyMPIComm_Get(py_comm.ptr());
     if (!comm) throw py::error_already_set();
     return comm;
}

void bind_dist_graph(py::module_& m)
{
    py::class_<DistGraph>(m, "DistGraph")
        .def(py::init([](py::object py_comm) { return DistGraph(*get_py_comm(py_comm)); }))
        .def("write_edge_file", &DistGraph::write_edge_file)
        .def("num_edges", &DistGraph::num_edges);
}

void bind_dist_point_vector(py::module_& m)
{
    py::class_<DistPointVector>(m, "DistPointVector")
        .def(py::init([](std::string fname, py::object py_comm) { return std::make_unique<DistPointVector>(fname.c_str(), *get_py_comm(py_comm)); }))
        .def("brute_force_systolic", [](const DistPointVector& points, Real radius, int verbosity)
                {
                    DistGraph graph(points.getcomm());
                    points.brute_force_systolic(radius, graph, verbosity);
                    return graph;
                }
            )
        .def("cover_tree_systolic", [](const DistPointVector& points, Real radius, Real cover, Index leaf_size, int verbosity)
                {
                    DistGraph graph(points.getcomm());
                    points.cover_tree_systolic(radius, cover, leaf_size, graph, verbosity);
                    return graph;
                }
            )
        .def("ghost_tree_voronoi", [](const DistPointVector& points, Real radius, Real cover, Index leaf_size, Index num_centers, std::string tree_assignment, std::string query_balancing, Index queries_per_tree, int verbosity)
                {
                    DistGraph graph(points.getcomm());
                    points.ghost_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment.c_str(), query_balancing.c_str(), queries_per_tree, graph, verbosity);
                    return graph;
                }
            )
        .def("cover_tree_voronoi", [](const DistPointVector& points, Real radius, Real cover, Index leaf_size, Index num_centers, std::string tree_assignment, std::string query_balancing, Index queries_per_tree, int verbosity)
                {
                    DistGraph graph(points.getcomm());
                    points.cover_tree_voronoi(radius, cover, leaf_size, num_centers, tree_assignment.c_str(), query_balancing.c_str(), queries_per_tree, graph, verbosity);
                    return graph;
                }
            )
        .def("totsize", [](const DistPointVector& points) { return points.gettotsize(); })
        .def("num_dimensions", &DistPointVector::num_dimensions)
        .def("dist_comps", [](const DistPointVector& points) { return points.dist_comps; })
        .def("my_comp_time", [](const DistPointVector& points) { return points.my_comp_time; })
        .def("my_comm_time", [](const DistPointVector& points) { return points.my_comm_time; });
}

PYBIND11_MODULE(metricspace, m)
{
    if (import_mpi4py() < 0) throw py::error_already_set();

    bind_dist_graph(m);
    bind_dist_point_vector(m);
}
