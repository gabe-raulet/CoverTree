#ifndef METRIC_H_
#define METRIC_H_

#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>
#include "bindutils.h"
#include "metricspace.h"

template <class Metric>
void bind_metric(py::module_& m, const std::string& name)
{
    using Index = typename Metric::index_type;
    using Real = typename Metric::real_type;
    using Atom = typename Metric::atom_type;

    using IndexVector = typename Metric::IndexVector;
    using RealVector = typename Metric::RealVector;

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
        .def("metric", [](const Metric& metric) { return std::string(metric.metric()); })
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
}

template <class Atom>
void bind_metrics(py::module_& m, const std::string& atom_name)
{
    using Index = int64_t;
    using Real = float;

    using euclidean = Euclidean<Index, Real, Atom>;
    using manhattan = Manhattan<Index, Real, Atom>;
    using chebyshev = Chebyshev<Index, Real, Atom>;

    bind_metric<euclidean>(m, std::string("EuclideanSpace") + atom_name);
    bind_metric<manhattan>(m, std::string("ManhattanSpace") + atom_name);
    bind_metric<chebyshev>(m, std::string("ChebyshevSpace") + atom_name);
}

#endif
