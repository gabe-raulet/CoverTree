#include "radius_neighbors_graph.h"
#include "dist_voronoi.h"
#include "dist_query.h"
#include "global_point.h"
#include "point_vector.h"
#include <numeric>

RadiusNeighborsGraph::RadiusNeighborsGraph(const char *filename, Real radius, MPI_Comm comm)
    : comm(comm),
      radius(radius),
      myptrs({0})
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    mypoints.read_fvecs(filename, myoffset, totsize, comm);
    mysize = mypoints.num_points();
}

void RadiusNeighborsGraph::write_graph_file(const char *filename) const
{
    std::ostringstream ss, ss2;
    Index num_edges, my_num_edges = 0;
    Index my_num_queries = myqueries.size();

    for (Index i = 0; i < my_num_queries; ++i)
        for (Index p = myptrs[i]; p < myptrs[i+1]; ++p)
            if (myqueries[i] != myneighs[p])
            {
                ss2 << (myqueries[i]+1) << " " << (myneighs[p]+1) << "\n";
                my_num_edges++;
            }

    for (Index i = 0; i < mysize; ++i)
    {
        ss2 << (i+myoffset+1) << " " << (i+myoffset+1) << "\n";
        my_num_edges++;
    }

    MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);

    if (!myrank) ss << "% " << totsize << " " << totsize << " " << num_edges << "\n" << ss2.str();
    else std::swap(ss, ss2);

    auto sbuf = ss.str();
    std::vector<char> buf(sbuf.begin(), sbuf.end());

    MPI_Offset mycount = buf.size(), fileoffset, filesize;
    MPI_Exscan(&mycount, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    int truncate = 0;

    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &filesize);
    truncate = (filesize > 0);
    MPI_Bcast(&truncate, 1, MPI_INT, 0, comm);
    if (truncate) MPI_File_set_size(fh, 0);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

struct BruteForceQuery
{
    Index operator()(const PointVector& mypoints, const AtomVector& curpoints, int dim, Index cursize, Index curoffset, Index mysize, Index myoffset, Real radius, IndexVector& myqueries, IndexVector& myneighs, IndexVector& myptrs)
    {
        Index num_edges = 0;

        for (Index j = 0; j < cursize; ++j)
        {
            myqueries.push_back(j+curoffset);

            for (Index i = 0; i < mysize; ++i)
                if (mypoints.distance(i, &curpoints[j*dim]) <= radius)
                {
                    myneighs.push_back(i+myoffset);
                    num_edges++;
                }

            myptrs.push_back(myneighs.size());
        }

        return num_edges;
    }
};

struct CoverTreeQuery
{
    CoverTree mytree;

    CoverTreeQuery(const PointVector& mypoints, Real cover, Index leaf_size)
    {
        mytree.build(mypoints, cover, leaf_size);
    }

    Index operator()(const PointVector& mypoints, const AtomVector& curpoints, int dim, Index cursize, Index curoffset, Index mysize, Index myoffset, Real radius, IndexVector& myqueries, IndexVector& myneighs, IndexVector& myptrs)
    {
        Index num_edges = 0;

        for (Index j = 0; j < cursize; ++j)
        {
            myqueries.push_back(j+curoffset);

            IndexVector neighs;
            num_edges += mytree.radius_query(mypoints, &curpoints[j*dim], radius, neighs);
            for (Index id : neighs) myneighs.push_back(id+myoffset);
            myptrs.push_back(myneighs.size());
        }

        return num_edges;
    }
};

Index RadiusNeighborsGraph::brute_force_systolic(int verbosity)
{
    BruteForceQuery indexer;
    return systolic(indexer);
}

Index RadiusNeighborsGraph::cover_tree_systolic(Real cover, Index leaf_size, int verbosity)
{
    CoverTreeQuery indexer(mypoints, cover, leaf_size);
    return systolic(indexer);
}

template <class Query>
Index RadiusNeighborsGraph::systolic(Query& indexer)
{
    IndexVector allsizes(nprocs), alloffsets(nprocs);
    allsizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, allsizes.data(), 1, MPI_INDEX, comm);

    std::exclusive_scan(allsizes.begin(), allsizes.end(), alloffsets.begin(), (Index)0);

    MPI_Request reqs[2];

    int next = (myrank+1)%nprocs;
    int prev = (myrank-1+nprocs)%nprocs;
    int cur = myrank;

    AtomVector curpoints = mypoints.copy_atoms();
    AtomVector nextpoints;

    int dim = mypoints.num_dimensions();

    Index num_edges = 0;

    for (int step = 0; step < nprocs; ++step)
    {
        Index numrecv = allsizes[(cur+1)%nprocs];
        Index numsend = allsizes[cur];

        int recvcount = numrecv*dim;
        int sendcount = numsend*dim;

        nextpoints.resize(recvcount);

        MPI_Irecv(nextpoints.data(), recvcount, MPI_ATOM, next, myrank, comm, &reqs[0]);
        MPI_Isend(curpoints.data(), sendcount, MPI_ATOM, prev, prev, comm, &reqs[1]);

        Index cursize = allsizes[cur];
        Index curoffset = alloffsets[cur];

        num_edges += indexer(mypoints, curpoints, dim, cursize, curoffset, mysize, myoffset, radius, myqueries, myneighs, myptrs);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        cur = (cur+1)%nprocs;
        std::swap(curpoints, nextpoints);
    }

    MPI_Allreduce(MPI_IN_PLACE, &num_edges, 1, MPI_INDEX, MPI_SUM, comm);

    return num_edges;
}

Index RadiusNeighborsGraph::cover_tree_voronoi(Real cover, Index leaf_size, Index num_centers, const char *tree_assignment, const char *query_balancing, int verbosity)
{
    double tottime, maxtime, mytime, t;

    MPI_Barrier(comm);
    tottime = -MPI_Wtime();

    /*
     * Partition points into Voronoi cells
     */

    mytime = -MPI_Wtime();
    DistVoronoi diagram(mypoints, 0, comm);
    diagram.add_next_centers(num_centers);
    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        Index mincellsize, maxcellsize;
        diagram.get_stats(mincellsize, maxcellsize, 0);

        if (!myrank) printf("[v1,time=%.3f] found %lld centers [separation=%.3f,minsize=%lld,maxsize=%lld,avgsize=%.3f]\n", maxtime, num_centers, diagram.center_separation(), mincellsize, maxcellsize, (totsize+0.0)/num_centers);
        fflush(stdout);
    }

    /*
     * Compute tree-to-rank assignments
     */

    Index s;
    IndexVector mycells;
    std::vector<int> dests; /* tree-to-rank assignments */

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    if      (!strcmp(tree_assignment, "static")) s = diagram.compute_static_cyclic_assignments(dests, mycells);
    else if (!strcmp(tree_assignment, "multiway")) s = diagram.compute_multiway_number_partitioning_assignments(dests, mycells);
    else throw std::runtime_error("invalid assignments_methods selected!");
    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) printf("[v1,time=%.3f] computed tree-to-rank assignments\n", maxtime);
        fflush(stdout);
    }

    /*
     * Gather cell points and find ghost points
     */

    MPI_Request reqs[2];
    MPI_Datatype MPI_GLOBAL_POINT;
    GlobalPoint::create_mpi_type(&MPI_GLOBAL_POINT, mypoints.num_dimensions());

    IndexVector mycellids, myghostids, mycellptrs, myghostptrs;

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    diagram.gather_local_cell_ids(mycellids, mycellptrs);
    diagram.gather_local_ghost_ids(radius, myghostids, myghostptrs);
    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        Index my_num_ghosts = myghostids.size(), num_ghosts;

        if (verbosity >= 2) { printf("[v2,rank=%d,time=%.3f] found %lld ghost points [cell_points=%lu]\n", myrank, mytime, my_num_ghosts, mycellids.size()); fflush(stdout); }

        MPI_Reduce(&my_num_ghosts, &num_ghosts, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) printf("[v1,time=%.3f] found %lld ghost points\n", maxtime, num_ghosts);
    }

    std::vector<int> cell_sendcounts, cell_recvcounts, cell_sdispls, cell_rdispls;
    std::vector<int> ghost_sendcounts, ghost_recvcounts, ghost_sdispls, ghost_rdispls;
    GlobalPointVector cell_sendbuf, cell_recvbuf;
    GlobalPointVector ghost_sendbuf, ghost_recvbuf;

    IndexVector my_query_sizes(s,0);
    std::vector<PointVector> my_cell_vectors(s, PointVector(mypoints.num_dimensions()));
    std::vector<IndexVector> my_cell_indices(s);

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    diagram.load_alltoall_outbufs(mycellids, mycellptrs, dests, cell_sendbuf, cell_sendcounts, cell_sdispls);
    diagram.load_alltoall_outbufs(myghostids, myghostptrs, dests, ghost_sendbuf, ghost_sendcounts, ghost_sdispls);
    global_point_alltoall(cell_sendbuf, cell_sendcounts, cell_sdispls, MPI_GLOBAL_POINT, cell_recvbuf, comm, &reqs[0]);
    global_point_alltoall(ghost_sendbuf, ghost_sendcounts, ghost_sdispls, MPI_GLOBAL_POINT, ghost_recvbuf, comm, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    MPI_Type_free(&MPI_GLOBAL_POINT);

    build_local_cell_vectors(cell_recvbuf, ghost_recvbuf, my_cell_vectors, my_cell_indices, my_query_sizes);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v1,time=%.3f] built local cell vectors\n", maxtime); fflush(stdout); }
    }

    /*
     * Build local cover trees
     */

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    std::vector<CoverTree> mytrees(s);

    for (Index i = 0; i < s; ++i)
    {
        t = -MPI_Wtime();
        mytrees[i].build(my_cell_vectors[i], cover, leaf_size);
        t += MPI_Wtime();

        if (verbosity >= 3) printf("[v3,rank=%d,time=%.3f] built cover tree [id=%lld,points=%lld,vertices=%lld]\n", myrank, t, mycells[i], my_cell_vectors[i].num_points(), mytrees[i].num_vertices());

        fflush(stdout);
    }

    mytime += MPI_Wtime();

    if (verbosity >= 2)
    {
        printf("[v2,rank=%d,time=%.3f] completed %lld local trees\n", myrank, mytime, s);
        fflush(stdout);
    }

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) printf("[v1,time=%.3f] built %lld cover trees\n", maxtime, num_centers);
        fflush(stdout);
    }

    /*
     * Compute epsilon neighbors
     */

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    DistQuery dist_query(mytrees, my_cell_vectors, my_cell_indices, my_query_sizes, mycells, radius, mypoints.num_dimensions(), comm, verbosity);

    if      (!strcmp(query_balancing, "static") || nprocs == 1) dist_query.static_balancing();
    else if (!strcmp(query_balancing, "steal")) dist_query.random_stealing(-1);
    else throw std::runtime_error("Invalid balancing_method selected!");

    mytime += MPI_Wtime();
    tottime += MPI_Wtime();

    Index edges;
    Index myedges = dist_query.my_edges_found();
    MPI_Allreduce(&myedges, &edges, 1, MPI_INDEX, MPI_SUM, comm);

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) printf("[v1,time=%.3f] completed queries [edges=%lld,density=%.3f]\n", mytime, edges, (edges+0.0)/totsize);
        fflush(stdout);
    }

    return edges;
}
