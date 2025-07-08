#ifndef BFORCE_H_
#define BFORCE_H_

#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>
#include "bindutils.h"
#include "metricspace.h"
#include "bruteforce.h"

template <class Metric>
void bind_brute_force(py::module_& m, const std::string& name)
{
    using Index = typename Metric::index_type;
    using Real = typename Metric::real_type;
    using Atom = typename Metric::atom_type;

    using IndexVector = typename Metric::IndexVector;
    using RealVector = typename Metric::RealVector;

    using bruteforce = BruteForce<Metric>;

    py::class_<bruteforce>(m, name.c_str())
        .def(py::init<const Metric&>())
        .def("num_points", &bruteforce::num_points)
        .def("num_dimensions", &bruteforce::num_dimensions)
        .def("radius_neighbors", [](const bruteforce& bf, NumpyArray<Atom>::type queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != bf.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       bf.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const bruteforce& bf, NumpyArray<Index>::type_flexible queries, Real radius, int num_threads)
                                   {
                                       Index num_queries = queries.shape()[0];
                                       Index dim = queries.shape()[1];

                                       if (dim != bf.num_dimensions()) throw std::runtime_error("Query dimension doesn't match request!");

                                       RealVector dists; IndexVector neighs, ptrs;
                                       bf.radius_neighbors(queries.data(), num_queries, radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("queries"), py::arg("radius"), py::arg("num_threads") = 1
            )
        .def("radius_neighbors", [](const bruteforce& bf, Real radius, int num_threads)
                                   {
                                       RealVector dists; IndexVector neighs, ptrs;
                                       bf.radius_neighbors(radius, dists, neighs, ptrs, num_threads);

                                       return std::make_tuple(dists, neighs, ptrs);

                                   }, py::arg("radius"), py::arg("num_threads") = 1
            );
}

template <class Atom>
void bind_brute_forces(py::module_& m, const std::string& atom_name)
{
    using Index = int64_t;
    using Real = float;

    using euclidean = Euclidean<Index, Real, Atom>;
    using manhattan = Manhattan<Index, Real, Atom>;
    using chebyshev = Chebyshev<Index, Real, Atom>;

    bind_brute_force<euclidean>(m, std::string("BruteForceEuclidean") + atom_name);
    bind_brute_force<manhattan>(m, std::string("BruteForceManhattan") + atom_name);
    bind_brute_force<chebyshev>(m, std::string("BruteForceChebyshev") + atom_name);
}

#endif
