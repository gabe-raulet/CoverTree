#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>

#include "utils.h"
#include "point_vector.h"
#include "cell_vector.h"
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

        Index m = diagram.num_centers();
        std::vector<int> dests(m);
        IndexVector mycells;
        Index s = 0;

        for (Index i = 0; i < m; ++i)
        {
            dests[i] = i % nprocs;

            if (dests[i] == myrank)
            {
                mycells.push_back(i);
                s++;
            }
        }

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

        MPI_Request reqs[2];

        MPI_Datatype MPI_GLOBAL_POINT;
        GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, points.num_dimensions());

        global_point_alltoall(cell_sendbuf, cell_sendcounts, cell_sdispls, MPI_GLOBAL_POINT, cell_recvbuf, MPI_COMM_WORLD, &reqs[0]);
        global_point_alltoall(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, MPI_GLOBAL_POINT, ghost_recvbuf, MPI_COMM_WORLD, &reqs[1]);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        MPI_Type_free(&MPI_GLOBAL_POINT);

        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!myrank) printf("[time=%.3f] alltoall exchange\n", maxtime);

        IndexVector cellcounts, ghostcounts;
        if (!myrank) { cellcounts.resize(nprocs); ghostcounts.resize(nprocs); }

        Index mycellcount = cell_recvbuf.size();
        Index myghostcount = ghost_recvbuf.size();

        MPI_Gather(&mycellcount, 1, MPI_INDEX, cellcounts.data(), 1, MPI_INDEX, 0, MPI_COMM_WORLD);
        MPI_Gather(&myghostcount, 1, MPI_INDEX, ghostcounts.data(), 1, MPI_INDEX, 0, MPI_COMM_WORLD);

        if (!myrank)
        {
            printf("cellcounts=%s\n", container_repr(cellcounts.begin(), cellcounts.end()).c_str());
            printf("ghostcounts=%s\n", container_repr(ghostcounts.begin(), ghostcounts.end()).c_str());
        }

        std::vector<CellVector> my_cell_vectors(s, CellVector(points.num_dimensions()));
        IndexVector my_query_sizes(s,0);
        build_local_cell_vectors(cell_recvbuf, ghost_recvbuf, my_cell_vectors, my_query_sizes);

        /* if (!myrank) */
        /* { */
            /* PointVector pts = my_cell_vectors.back(); */
            /* CoverTree tree; */
            /* tree.build(pts, 1.3, 20); */
            /* printf("%lld\n", pts.num_points()); */

            /* IndexVector neighs; */
            /* tree.radius_query(pts, pts[0], radius, neighs); */
            /* printf("%s\n", container_repr(neighs.begin(), neighs.end()).c_str()); */

            /* neighs.clear(); */
            /* tree.radius_query(my_cell_vectors.back(), my_cell_vectors.back()[0], radius, neighs); */
            /* printf("%s\n", container_repr(neighs.begin(), neighs.end()).c_str()); */
        /* } */
    }

    MPI_Finalize();
    return 0;
}
