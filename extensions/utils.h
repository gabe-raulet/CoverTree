#ifndef UTILS_H_
#define UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <class T>
static inline T* npmem(py::array_t<T> numpy_array)
{
    py::buffer_info info = numpy_array.request();
    return static_cast<T*>(info.ptr);
}

#endif
