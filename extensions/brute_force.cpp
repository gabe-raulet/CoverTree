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

class brute_force
{
    private:

        int64_t n, d;
        std::string metric;

    public:

        brute_force(py::array_t<float> points_py, std::string metric) : metric(metric)
        {
            py::buffer_info info = points_py.request();
            n = info.shape[0], d = info.shape[1];
        }

        std::string get_metric() const { return metric; }

        template <class DistanceFunctor>
        std::tuple<std::vector<float>, std::vector<int64_t>, std::vector<int64_t>>
        radius_neighbors_graph(const float *points, float radius, int num_threads) const
        {
            static DistanceFunctor distance;

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
    py::class_<brute_force>(m, "brute_force")
        .def(py::init<py::array_t<float>, std::string>())
        .def("radius_neighbors_graph",
                [](const brute_force& bf, py::array_t<float> points_py, float radius, int num_threads)
                  {
                      std::string metric = bf.get_metric();

                      if (metric == "euclidean" || metric == "l2") return bf.radius_neighbors_graph<EuclideanDistance>(npmem(points_py), radius, num_threads);
                      else if (metric == "manhattan" || metric == "l1") return bf.radius_neighbors_graph<ManhattanDistance>(npmem(points_py), radius, num_threads);
                      else if (metric == "chebyshev" || metric == "infinity") return bf.radius_neighbors_graph<ChebyshevDistance>(npmem(points_py), radius, num_threads);
                      else if (metric == "cosine") return bf.radius_neighbors_graph<CosineDistance>(npmem(points_py), radius, num_threads);
                      else if (metric == "angular") return bf.radius_neighbors_graph<AngularDistance>(npmem(points_py), radius, num_threads);
                      else throw std::runtime_error("Invalid metric!");
                  }
            );
        /* .def("radius_neighbors_graph", [](const brute_force& bf, py::array_t<float> points_py, float radius, int num_threads) { return bf.radius_neighbors_graph(npmem(points_py), radius, num_threads); }); */
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
