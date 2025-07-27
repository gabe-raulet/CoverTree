#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>

#include "utils.h"
#include "point_vector.h"
#include "dist_voronoi.h"

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 4)
    {
        if (!myrank) fprintf(stderr, "Usage: %s <points> <centers> <radius>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    const char *fname = argv[1];
    Index centers = atoi(argv[2]);
    Real radius = atof(argv[3]);

    PointVector points; points.read_fvecs(fname, MPI_COMM_WORLD);

    {
        double t, maxtime;

        t = -MPI_Wtime();
        DistVoronoi diagram(points, 0, MPI_COMM_WORLD);
        diagram.add_next_centers(centers);
        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] build diagram %s\n", maxtime, diagram.repr().c_str());

        IndexVector mycellids, myghostids, mycellptrs, myghostptrs;

        t = -MPI_Wtime();
        diagram.gather_local_cell_ids(mycellids, mycellptrs);
        diagram.gather_local_ghost_ids(radius, myghostids, myghostptrs);
        t += MPI_Wtime();

        Index num_ghosts = myghostids.size();
        MPI_Allreduce(MPI_IN_PLACE, &num_ghosts, 1, MPI_INDEX, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] find ghost points [num_ghosts=%lld]\n", maxtime, num_ghosts);

        std::vector<int> dests(diagram.num_centers());

        for (int i = 0; i < dests.size(); ++i)
            dests[i] = i % nprocs;

        std::vector<int> cell_sendcounts, cell_recvcounts, cell_sdispls, cell_rdispls;
        std::vector<int> ghost_sendcounts, ghost_recvcounts, ghost_sdispls, ghost_rdispls;

        std::vector<GlobalPoint> cell_sendbuf, cell_recvbuf;
        std::vector<GlobalPoint> ghost_sendbuf, ghost_recvbuf;

        t = -MPI_Wtime();
        diagram.load_alltoall_outbufs(mycellids, mycellptrs, dests, cell_sendbuf, cell_sendcounts, cell_sdispls);
        diagram.load_alltoall_outbufs(myghostids, myghostptrs, dests, ghost_sendbuf, ghost_sendcounts, ghost_sdispls);
        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] loaded alltoall outbufs\n", maxtime);

        t = -MPI_Wtime();

        cell_recvcounts.resize(nprocs), cell_rdispls.resize(nprocs);
        ghost_recvcounts.resize(nprocs), ghost_rdispls.resize(nprocs);

        MPI_Datatype MPI_GLOBAL_POINT;
        GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, points.num_dimensions());

        MPI_Request reqs[2];

        MPI_Ialltoall(cell_sendcounts.data(), 1, MPI_INT, cell_recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD, reqs);
        MPI_Ialltoall(ghost_sendcounts.data(), 1, MPI_INT, ghost_recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD, reqs+1);

        MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        std::exclusive_scan(cell_recvcounts.begin(), cell_recvcounts.end(), cell_rdispls.begin(), static_cast<int>(0));
        cell_recvbuf.resize(cell_recvcounts.back()+cell_rdispls.back());

        MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        std::exclusive_scan(ghost_recvcounts.begin(), ghost_recvcounts.end(), ghost_rdispls.begin(), static_cast<int>(0));
        ghost_recvbuf.resize(ghost_recvcounts.back()+ghost_rdispls.back());

        MPI_Ialltoallv(cell_sendbuf.data(), cell_sendcounts.data(), cell_sdispls.data(), MPI_GLOBAL_POINT,
                       cell_recvbuf.data(), cell_recvcounts.data(), cell_rdispls.data(), MPI_GLOBAL_POINT, MPI_COMM_WORLD, reqs);

        MPI_Ialltoallv(ghost_sendbuf.data(), ghost_sendcounts.data(), ghost_sdispls.data(), MPI_GLOBAL_POINT,
                       ghost_recvbuf.data(), ghost_recvcounts.data(), ghost_rdispls.data(), MPI_GLOBAL_POINT, MPI_COMM_WORLD, reqs+1);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        MPI_Type_free(&MPI_GLOBAL_POINT);

        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] alltoall exchange\n", maxtime);
    }

    MPI_Finalize();
    return 0;
}
