#include "global_point.h"
#include <assert.h>
#include <numeric>

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

void global_point_alltoall(const GlobalPointVector& sendbuf, const std::vector<int>& sendcounts, const std::vector<int>& sdispls, MPI_Datatype MPI_GLOBAL_POINT, GlobalPointVector& recvbuf, MPI_Comm comm, MPI_Request *request)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    std::vector<int> recvcounts(nprocs), rdispls(nprocs);

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    recvbuf.resize(recvcounts.back()+rdispls.back());

    MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_GLOBAL_POINT,
                   recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_GLOBAL_POINT, comm, request);
}

void build_local_cell_vectors(const GlobalPointVector& my_cell_points, const GlobalPointVector& my_ghost_points, std::vector<CellVector>& my_cell_vectors, IndexVector& my_query_sizes)
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
    }

    for (const auto& p : my_cell_points) my_cell_vectors[p.cell].push_back(p);
    for (const auto& p : my_ghost_points) my_cell_vectors[p.cell].push_back(p);
}
