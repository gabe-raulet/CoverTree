#ifndef BIND_UTILS_H_
#define BIND_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>

namespace py = pybind11;

template <class T>
struct NumpyArray
{
    using type = py::array_t<T, py::array::c_style>;
    using type_flexible = py::array_t<T, py::array::c_style | py::array::forcecast>;
};

#endif
