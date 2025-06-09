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

class BruteForce
{
    public:

        BruteForce(py::array_t<double> points_py)
        {
            py::buffer_info info = points_py.request();
            num_points = info.shape[0];
            dim = info.shape[1];

            double *data = static_cast<double*>(info.ptr);
            points.assign(data, data + num_points*dim);
        }

        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>>
        radius_neighbors_graph(double radius, int num_threads)
        {
            static EuclideanDistance distance;

            omp_set_num_threads(num_threads);

            int64_t n = num_points;
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
                        double dist = distance(&points[j*dim], &points[i*dim], dim);

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

            return std::make_tuple(rowptrs, colids, dists);
        }

    private:

        int64_t num_points, dim;
        std::vector<double> points;
};

PYBIND11_MODULE(brute_force, m)
{
    py::class_<BruteForce>(m, "_BruteForce")
        .def(py::init<py::array_t<double>>())
        .def("_radius_neighbors_graph", &BruteForce::radius_neighbors_graph);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++20', '-fopenmp', '-O3']
%>
*/
