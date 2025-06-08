//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
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

void brute_force_knn_query(double *P, double *q, double *dists, int64_t *inds, int64_t n, int64_t d, int64_t k)
{
    static EuclideanDistance distance;

    std::vector<std::tuple<double, int64_t>> results(n);

    for (int64_t i = 0; i < n; ++i)
    {
        std::get<0>(results[i]) = distance(&P[i*d], q, d);
        std::get<1>(results[i]) = i;
    }

    std::sort(results.begin(), results.end());

    for (int64_t i = 0; i < k; ++i)
    {
        dists[i] = std::get<0>(results[i]);
        inds[i] = std::get<1>(results[i]);
    }
}

void brute_force_query_(double *P, double *X, double *dists, int64_t *inds, int64_t n, int64_t d, int64_t m, int64_t k)
{
    for (uint64_t i = 0; i < m; ++i)
    {
        brute_force_knn_query(P, &X[i*d], &dists[i*k], &inds[i*k], n, d, k);
    }
}

void brute_force_query(py::array_t<double> P_py, py::array_t<double> X_py, py::array_t<double> dists_py, py::array_t<int64_t> inds_py, int num_threads)
{
    py::buffer_info Pinfo = P_py.request();
    py::buffer_info Xinfo = X_py.request();
    py::buffer_info distsinfo = dists_py.request();
    py::buffer_info indsinfo = inds_py.request();

    int64_t n = Pinfo.shape[0];
    int64_t d = Pinfo.shape[1];

    int64_t m = Xinfo.shape[0];
    int64_t k = distsinfo.shape[1];

    if (d != Xinfo.shape[1]) throw std::runtime_error("error: P.shape[1] != X.shape[1]");
    else if (m != distsinfo.shape[0]) throw std::runtime_error("error: X.shape[0] != dists.shape[0] (m)");
    else if (m != indsinfo.shape[0]) throw std::runtime_error("error: X.shape[0] != inds.shape[0] (m)");
    else if (k != indsinfo.shape[1]) throw std::runtime_error("error: dists.shape[1] != inds.shape[1] (k)");
    else if (k > n) throw std::runtime_error("error: k > P.shape[0]");

    double *P = static_cast<double*>(Pinfo.ptr);
    double *X = static_cast<double*>(Xinfo.ptr);
    double *dists = static_cast<double*>(distsinfo.ptr);
    int64_t *inds = static_cast<int64_t*>(indsinfo.ptr);

    omp_set_num_threads(1);

    brute_force_query_(P, X, dists, inds, n, d, m, k);
}


PYBIND11_MODULE(brute_force, m)
{
    m.def("distance_matrix", &distance_matrix, py::arg("P_py"), py::arg("num_threads") = 4);
    m.def("brute_force_query", &brute_force_query);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
