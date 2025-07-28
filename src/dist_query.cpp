#include "dist_query.h"
#include <random>

Index GhostTree::make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made)
{
    Index edges_found = 0;

    if (finished()) return edges_found;

    if (count < 0) count = num_queries - cur_query;
    else count = std::min(count, num_queries - cur_query);

    queries_made = count;

    for (Index i = 0; i < count; ++i, ++cur_query)
    {
        Index found = tree.radius_query(points, cur_query, radius, neighs);
        queries.push_back(points.index(cur_query));
        ptrs.push_back(neighs.size());
        edges_found += found;
    }

    return edges_found;
}

DistQuery::DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<CellVector>& my_cell_vectors, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, MPI_Comm comm, int verbosity)
    : radius(radius),
      comm(comm),
      myptrs({0}),
      num_local_trees_completed(0),
      num_local_queries_made(0),
      num_local_edges_found(0),
      verbosity(verbosity)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index s = mytrees.size();

    for (Index i = 0; i < s; ++i)
    {
        myqueue.emplace_back(mytrees[i], my_cell_vectors[i], my_query_sizes[i], mycells[i]);
    }

    MPI_Allreduce(&s, &num_global_trees, 1, MPI_INDEX, MPI_SUM, comm);
}

void DistQuery::static_balancing()
{
    double t;

    t = -MPI_Wtime();
    for (auto& tree : myqueue) make_tree_queries(tree, -1);
    t += MPI_Wtime();

    if (verbosity > 1) report_finished(t);
}

bool DistQuery::make_tree_queries(GhostTree& tree, Index count)
{
    if (tree.finished())
        return true;

    double t;
    Index queries_made, edges_found;

    t = -MPI_Wtime();
    edges_found = tree.make_queries(count, radius, myneighs, myqueries, myptrs, queries_made);
    t += MPI_Wtime();

    num_local_queries_made += queries_made;
    num_local_edges_found += edges_found;

    if (verbosity > 2) { printf("[v3,rank=%d,time=%.3f] queried ghost tree [id=%lld,queries_made=%lld,queries_left=%lld,edges_found=%lld]\n", myrank, t, tree.id, queries_made, tree.num_queries - tree.cur_query, edges_found); fflush(stdout); }

    if (tree.finished())
    {
        num_local_trees_completed++;
        return true;
    }
    else
    {
        return false;
    }
}

void DistQuery::report_finished(double mytime)
{
    Real density = (num_local_edges_found+0.0)/num_local_queries_made;
    printf("[v2,rank=%d,time=%.3f] completed queries [num_local_trees=%lld,num_total_queries=%lld,num_local_edges=%lld,density=%.3f]\n", myrank, mytime, num_local_trees_completed, num_local_queries_made, num_local_edges_found, density);
    fflush(stdout);
}

void DistQuery::write_to_file(const char *fname) const
{
    std::ostringstream ss;
    Index num_vertices, num_edges;

    MPI_Reduce(&num_local_edges_found, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);
    MPI_Reduce(&num_local_queries_made, &num_vertices, 1, MPI_INDEX, MPI_SUM, 0, comm);

    if (!myrank) ss << num_vertices << " " << num_vertices << " " << num_edges << "\n";

    for (Index i = 0; i < num_local_queries_made; ++i)
        for (Index p = myptrs[i]; p < myptrs[i+1]; ++p)
            ss << (myqueries[i]+1) << " " << (myneighs[p]+1) << "\n";

    auto sbuf = ss.str();
    std::vector<char> buf(sbuf.begin(), sbuf.end());

    MPI_Offset mysize = buf.size(), fileoffset;
    MPI_Exscan(&mysize, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    MPI_File fh;
    MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

/* void DistQuery::shuffle_queues() */
/* { */
    /* static std::random_device rd; */
    /* static std::default_random_engine gen(rd()); */
    /* std::uniform_int_distribution<int> dist{0,nprocs-1}; */

    /* int num_trees_send = myqueue.size(); */

    /* std::vector<int> dests(num_trees_send); */
    /* std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs); */
    /* std::vector<GhostTree> sendbuf(num_trees_send), recvbuf; */

    /* for (int i = 0; i < num_trees_send; ++i) */
    /* { */
        /* dests[i] = dist(gen); */
        /* sendcounts[dests[i]]++; */
    /* } */

    /* std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0)); */
    /* auto ptrs = sdispls; */

    /* for (int i = 0; i < num_trees_send; ++i) */
    /* { */
        /* sendbuf[ptrs[dests[i]]++] = myqueue[i]; */
    /* } */

    /* MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm); */

    /* std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0)); */
    /* recvbuf.resize(recvcounts.back()+rdispls.back()); */
/* } */
