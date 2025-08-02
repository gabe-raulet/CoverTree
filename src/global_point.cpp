#include "global_point.h"
#include <assert.h>
#include <numeric>
#include <algorithm>

void GlobalPoint::create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT, int dim)
{
    assert((dim <= MAX_DIM));

    int blklens[4] = {dim,1,1,1};
    MPI_Aint disps[4] = {offsetof(GlobalPoint,p), offsetof(GlobalPoint,id), offsetof(GlobalPoint,cell), offsetof(GlobalPoint,dist)};
    MPI_Datatype types[4] = {MPI_ATOM, MPI_INDEX, MPI_INDEX, MPI_REAL};
    MPI_Type_create_struct(4, blklens, disps, types, MPI_GLOBAL_POINT);
    MPI_Type_commit(MPI_GLOBAL_POINT);
}

GlobalPoint& GlobalPoint::operator=(const GlobalPoint& rhs)
{
    id = rhs.id;
    cell = rhs.cell;
    dist = rhs.dist;
    std::copy(rhs.p, rhs.p+MAX_DIM, p);
    return *this;
}

std::string GlobalPoint::repr() const
{
    char buf[512];
    snprintf(buf, 512, "GlobalPoint(id=%lld,cell=%lld,dist=%.3f)", id, cell, dist);
    return std::string(buf);
}

void build_local_cell_vectors(const GlobalPointVector& my_cell_points, const GlobalPointVector& my_ghost_points, std::vector<PointVector>& my_cell_vectors, std::vector<IndexVector>& my_cell_indices, IndexVector& my_query_sizes)
{
    Index s = my_cell_vectors.size();
    assert((s == my_query_sizes.size()));

    std::fill(my_query_sizes.begin(), my_query_sizes.end(), 0);
    IndexVector my_vector_sizes(s, 0);

    for (const auto& [p, id, cell, dist] : my_cell_points) { my_query_sizes[cell]++; my_vector_sizes[cell]++; }
    for (const auto& [p, id, cell, dist] : my_ghost_points) { my_vector_sizes[cell]++; }

    for (Index i = 0; i < s; ++i)
    {
        my_cell_vectors[i].clear();
        my_cell_vectors[i].reserve(my_vector_sizes[i]);
        my_cell_indices[i].reserve(my_vector_sizes[i]);
    }

    /* std::sort(my_cell_points.begin(), my_cell_points.end(), [](const auto& lhs, const auto& rhs) { return lhs.dist < rhs.dist; }); */

    for (const auto& p : my_cell_points) { my_cell_vectors[p.cell].push_back(p.p); my_cell_indices[p.cell].push_back(p.id); }
    for (const auto& p : my_ghost_points) { my_cell_vectors[p.cell].push_back(p.p); my_cell_indices[p.cell].push_back(p.id); }
}
