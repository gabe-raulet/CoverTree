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

template <class T>
static inline T* npmem(py::array_t<T> numpy_array)
{
    py::buffer_info info = numpy_array.request();
    return static_cast<T*>(info.ptr);
}

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

class BruteForce
{
    private:

        int64_t n, d;

    public:

        BruteForce(py::array_t<double> points_py)
        {
            py::buffer_info info = points_py.request();
            n = info.shape[0], d = info.shape[1];
        }

        std::tuple<std::vector<double>, std::vector<int64_t>, std::vector<int64_t>>
        radius_neighbors_graph(const double *points, double radius, int num_threads) const
        {
            static EuclideanDistance distance;

            omp_set_num_threads(num_threads);

            std::vector<int64_t> rowptrs(n+1), w(n,0), colids;
            std::vector<double> dists;

            std::vector<std::tuple<int64_t, int64_t, double>> tuples;
            int64_t *wptr = w.data();

            #pragma omp parallel shared(w, tuples)
            {
                std::vector<std::tuple<int64_t, int64_t, double>> mytuples;

                #pragma omp for nowait reduction(+:wptr[:n])
                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        double dist = distance(&points[j*d], &points[i*d], d);

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
    py::class_<BruteForce>(m, "brute_force")
        .def(py::init<py::array_t<double>>())
        .def("radius_neighbors_graph", [](const BruteForce& bf, py::array_t<double> points_py, double radius, int num_threads) { return bf.radius_neighbors_graph(npmem(points_py), radius, num_threads); });
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
