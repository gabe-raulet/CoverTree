//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <omp.h>

namespace py = pybind11;

struct EuclideanDistance
{
    double operator()(const double *p, const double *q, int d)
    {
        double val = 0;

        for (int i = 0; i < d; ++i)
        {
            double delta = p[i] - q[i];
            val += delta * delta;
        }

        return std::sqrt(val);
    }
};

py::array_t<double> distance_matrix(py::array_t<double> P_py, int num_threads = 4)
{
    py::buffer_info Pinfo = P_py.request();
    uint64_t n = Pinfo.shape[0];
    int d = Pinfo.shape[1];

    double *P = static_cast<double*>(Pinfo.ptr);
    py::array_t<double> D_py({n, n});
    auto D = D_py.mutable_unchecked<2>();

    EuclideanDistance distance;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t j = i; j < n; ++j)
            D(i,j) = D(j,i) = distance(&P[i*d], &P[j*d], d);

    return D_py;
}


PYBIND11_MODULE(brute_force, m)
{
    m.def("distance_matrix", &distance_matrix, py::arg("P_py"), py::arg("num_threads") = 4);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
