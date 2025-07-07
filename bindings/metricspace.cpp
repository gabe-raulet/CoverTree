#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include "bruteforce.h"
#include "metricspace/metricspace.h"

namespace py = pybind11;

template <class T>
struct NumpyArray
{
    using type = py::array_t<T, py::array::c_style>;
    using type_flexible = py::array_t<T, py::array::c_style | py::array::forcecast>;
};

template <class Metric>
void bind_metric(py::module_& m, const std::string& name)
{
    using Index = typename Metric::index_type;
    using Real = typename Metric::real_type;
    using Atom = typename Metric::atom_type;

    using IndexVector = typename Metric::IndexVector;
    using RealVector = typename Metric::RealVector;
    using TripleVector = typename Metric::TripleVector;

    py::class_<Metric>(m, name.c_str(), py::buffer_protocol())
        .def(py::init([](NumpyArray<Atom>::type array) { return Metric(array.data(), array.shape()[0], array.shape()[1]); }))
        .def("distance", [](const Metric& metric, Index p, Index q)
                           {
                               return metric.distance(p, q);
                           }
            )
        .def("distance", [](const Metric& metric, Index p, py::buffer q_buf)
                           {
                               py::buffer_info q_info = q_buf.request();

                               if (q_info.format != py::format_descriptor<Atom>::format() || q_info.ndim != 1)
                                   throw std::runtime_error("Incompatible buffer format!");

                               const Atom *q = static_cast<const Atom*>(q_info.ptr);

                               return metric.distance(p, q);
                           }
            )
        .def("distance", [](const Metric& metric, py::buffer p_buf, py::buffer q_buf)
                           {
                               py::buffer_info p_info = p_buf.request();
                               py::buffer_info q_info = q_buf.request();

                               if (p_info.format != py::format_descriptor<Atom>::format() || p_info.ndim != 1)
                                   throw std::runtime_error("Incompatible buffer format!");

                               if (q_info.format != py::format_descriptor<Atom>::format() || q_info.ndim != 1)
                                   throw std::runtime_error("Incompatible buffer format!");

                               const Atom *p = static_cast<const Atom*>(p_info.ptr);
                               const Atom *q = static_cast<const Atom*>(q_info.ptr);

                               return metric.distance(p, q);
                           }
            )
        .def("num_points", &Metric::num_points)
        .def("num_dimensions", &Metric::num_dimensions)
        .def_buffer([](Metric& metric) -> py::buffer_info
                      {
                          void *ptr = metric.data();
                          py::ssize_t itemsize = sizeof(Atom);
                          std::string format = py::format_descriptor<Atom>::format();
                          py::ssize_t ndim = 2;
                          py::ssize_t n = metric.num_points(), d = metric.num_dimensions();
                          std::vector<py::ssize_t> shape = {n, d};
                          std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(Atom)*d), static_cast<py::ssize_t>(sizeof(Atom))};

                          return py::buffer_info(ptr, itemsize, format, ndim, shape, strides);
                      }
                   );

    std::string bf_name = std::string("BruteForce") + name;

    py::class_<BruteForce<Metric>>(m, bf_name.c_str())
        .def(py::init<const Metric&>())
        .def("num_points", &BruteForce<Metric>::num_points)
        .def("num_dimensions", &BruteForce<Metric>::num_dimensions)
        .def("radius_query", [](const BruteForce<Metric>& bf, Index query, Real radius, bool return_distance) -> py::object
                               {
                                   IndexVector neighs; RealVector dists;
                                   bf.radius_query(query, radius, neighs, dists);

                                   if (return_distance) return py::cast(std::make_tuple(dists, neighs));
                                   else return py::cast(neighs);
                               }, py::arg("query"), py::arg("radius"), py::arg("return_distance") = true
            )
        .def("radius_neighbors", [](const BruteForce<Metric>& bf, NumpyArray<Atom>::type queries, Real radius, bool return_distance, int num_threads) -> py::object
                                   {
                                       IndexVector neighs, ptrs; RealVector dists;
                                       Index num_queries = queries.shape()[0];
                                       if (queries.shape()[1] != bf.num_dimensions()) throw std::runtime_error("Incompatible buffer format!");
                                       bf.radius_neighbors(queries.data(), num_queries, radius, neighs, dists, ptrs, num_threads);
                                       if (return_distance) return py::cast(std::make_tuple(dists, neighs, ptrs));
                                       else return py::cast(std::make_tuple(neighs, ptrs));
                                   }, py::arg("queries"), py::arg("radius"), py::arg("return_distance") = true, py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const BruteForce<Metric>& bf, NumpyArray<Index>::type_flexible queries, Real radius, bool return_distance, int num_threads) -> py::object
                                   {
                                       IndexVector neighs, ptrs; RealVector dists;
                                       Index num_queries = queries.shape()[0];
                                       bf.radius_neighbors(IndexVector(queries.data(), queries.data() + num_queries), radius, neighs, dists, ptrs, num_threads);
                                       if (return_distance) return py::cast(std::make_tuple(dists, neighs, ptrs));
                                       else return py::cast(std::make_tuple(neighs, ptrs));
                                   }, py::arg("queries"), py::arg("radius"), py::arg("return_distance") = true, py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const BruteForce<Metric>& bf, Real radius, bool return_distance, int num_threads) -> py::object
                                   {
                                       IndexVector neighs, ptrs; RealVector dists;
                                       bf.radius_neighbors(radius, neighs, dists, ptrs, num_threads);
                                       if (return_distance) return py::cast(std::make_tuple(dists, neighs, ptrs));
                                       else return py::cast(std::make_tuple(neighs, ptrs));
                                   }, py::arg("radius"), py::arg("return_distance") = true, py::arg("num_threads") = 1
            )
        .def("radius_neighbors_dist", [](const BruteForce<Metric>& bf, Real radius, py::object py_comm, bool return_distance) -> py::object
                                   {
                                       MPI_Comm *comm_ptr = PyMPIComm_Get(py_comm.ptr());

                                       if (!comm_ptr) throw py::error_already_set();

                                       IndexVector neighs, ptrs; RealVector dists;
                                       bf.radius_neighbors(radius, neighs, dists, ptrs, *comm_ptr);
                                       if (return_distance) return py::cast(std::make_tuple(dists, neighs, ptrs));
                                       else return py::cast(std::make_tuple(neighs, ptrs));
                                   }, py::arg("radius"), py::arg("py_comm"), py::arg("return_distance") = true
            );
}

template <class Atom>
void bind_metrics(py::module_& m, const std::string& atom_name)
{
    using Index = int64_t;
    using Real = float;

    using euclidean = Euclidean<Index, Real, Atom>;
    using levenshtein = Levenshtein<Index, Real, Atom>;

    bind_metric<euclidean>(m, std::string("EuclideanSpace") + atom_name);
    bind_metric<levenshtein>(m, std::string("LevenshteinSpace") + atom_name);
}

PYBIND11_MODULE(metricspace, m)
{
    if (import_mpi4py() < 0) throw py::error_already_set();

    bind_metrics<float>(m, "Float");
    bind_metrics<double>(m, "Double");
}
