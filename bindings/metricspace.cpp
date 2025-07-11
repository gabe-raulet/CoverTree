#include "metric.h"
#include "bforce.h"
#include "ctree.h"

PYBIND11_MODULE(metricspace, m)
{
    if (import_mpi4py() < 0) throw py::error_already_set();

    bind_metrics<float>(m, "Float");
    bind_metrics<double>(m, "Double");
    bind_metrics<uint8_t>(m, "UChar");

    bind_brute_forces<float>(m, "Float");
    bind_brute_forces<double>(m, "Double");
    bind_brute_forces<uint8_t>(m, "UChar");

    bind_cover_trees<float>(m, "Float");
    bind_cover_trees<uint8_t>(m, "UChar");
}
