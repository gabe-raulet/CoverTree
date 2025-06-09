//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include "utils.h"
#include "metrics.h"

namespace py = pybind11;

template <class Distance>
class brute_force
{
    static inline constexpr Distance distance = Distance();

    private:

        int64_t n, d;

    public:

        brute_force(py::array_t<float> points_py)
        {
            py::buffer_info info = points_py.request();
            n = info.shape[0], d = info.shape[1];
        }

        std::tuple<std::vector<float>, std::vector<int64_t>, std::vector<int64_t>>
        radius_neighbors_graph(py::array_t<float> points_py, float radius, int num_threads) const
        {
            py::buffer_info info = points_py.request();
            float *points = static_cast<float*>(info.ptr);

            omp_set_num_threads(num_threads);

            std::vector<int64_t> rowptrs(n+1), w(n,0), colids;
            std::vector<float> dists;

            std::vector<std::tuple<int64_t, int64_t, float>> tuples;
            int64_t *wptr = w.data();

            #pragma omp parallel shared(w, tuples)
            {
                std::vector<std::tuple<int64_t, int64_t, float>> mytuples;

                #pragma omp for nowait reduction(+:wptr[:n])
                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        float dist = distance(&points[j*d], &points[i*d], d);

                        if (dist <= radius)
                        {
                            mytuples.emplace_back(i, j, dist);
                            w[i]++;
                        }
                    }
                }

                #pragma omp critical
                {
                    std::copy(mytuples.begin(), mytuples.end(), std::back_inserter(tuples));
                }
            }

            int64_t nz = 0;

            for (int64_t i = 0; i < n; ++i)
            {
                rowptrs[i] = nz;
                nz += w[i];
                w[i] = rowptrs[i];
            }

            rowptrs[n] = nz;
            colids.resize(nz);
            dists.resize(nz);

            for (const auto& [i, j, dist] : tuples)
            {
                int64_t p;
                colids[p = w[i]++] = j;
                dists[p] = dist;
            }

            return std::make_tuple(dists, colids, rowptrs);
        }
};

PYBIND11_MODULE(brute_force, m)
{
    py::class_<brute_force<EuclideanDistance>>(m, "brute_force_euclidean")
        .def(py::init<py::array_t<float>>())
        .def("radius_neighbors_graph",  &brute_force<EuclideanDistance>::radius_neighbors_graph);

    py::class_<brute_force<ManhattanDistance>>(m, "brute_force_manhattan")
        .def(py::init<py::array_t<float>>())
        .def("radius_neighbors_graph",  &brute_force<ManhattanDistance>::radius_neighbors_graph);

    py::class_<brute_force<ChebyshevDistance>>(m, "brute_force_chebyshev")
        .def(py::init<py::array_t<float>>())
        .def("radius_neighbors_graph",  &brute_force<ChebyshevDistance>::radius_neighbors_graph);

    py::class_<brute_force<AngularDistance>>(m, "brute_force_angular")
        .def(py::init<py::array_t<float>>())
        .def("radius_neighbors_graph",  &brute_force<AngularDistance>::radius_neighbors_graph);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
