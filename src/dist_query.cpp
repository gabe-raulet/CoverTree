#include "dist_query.h"
#include "work_stealer.h"
#include <assert.h>
#include <random>
#include <algorithm>

DistQuery::DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<PointVector>& my_cell_vectors, const std::vector<IndexVector>& my_cell_indices, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, int dim, MPI_Comm comm, int verbosity)
    : radius(radius),
      comm(comm),
      myptrs({0}),
      num_local_trees_completed(0),
      num_local_queries_made(0),
      num_local_edges_found(0),
      dim(dim),
      verbosity(verbosity)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index s = mytrees.size();

    for (Index i = 0; i < s; ++i)
    {
        myqueue.emplace_back(mytrees[i], my_cell_vectors[i], my_cell_indices[i], my_query_sizes[i], mycells[i]);
    }

    MPI_Allreduce(&s, &num_global_trees, 1, MPI_INDEX, MPI_SUM, comm);
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

    if (verbosity > 2) { printf("[v3,rank=%d,time=%.3f] queried ghost tree [id=%lld,queries_made=%lld,queries_left=%lld,edges_found=%lld]\n", myrank, t, tree.header.id, queries_made, tree.header.num_queries - tree.header.cur_query, edges_found); fflush(stdout); }

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
    printf("[v2,rank=%d,time=%.3f] completed queries [num_local_trees=%lld,num_total_queries=%lld,num_local_edges=%lld,density=%.3f]\n", myrank, mytime, num_local_trees_completed, num_local_queries_made, num_local_edges_found, density); fflush(stdout);
}

void DistQuery::report_finished(double my_comp_time, double my_steal_time, double my_poll_time, double my_response_time, double my_allreduce_time)
{
    Real density = (num_local_edges_found+0.0)/num_local_queries_made;
    printf("[v2,rank=%d,comp_time=%.3f,steal_time=%.3f,poll_time=%.3f,resp_time=%.3f,iallreduce_time=%.3f] completed queries [num_local_trees=%lld,num_total_queries=%lld,num_local_edges=%lld,density=%.3f]\n", myrank, my_comp_time, my_steal_time, my_poll_time, my_response_time, my_allreduce_time, num_local_trees_completed, num_local_queries_made, num_local_edges_found, density); fflush(stdout);
}

void DistQuery::write_to_file(const char *fname) const
{
    std::ostringstream ss, ss2;
    Index num_vertices, num_edges, my_num_edges = 0;

    for (Index i = 0; i < num_local_queries_made; ++i)
        for (Index p = myptrs[i]; p < myptrs[i+1]; ++p)
            if (myqueries[i] != myneighs[p])
            {
                ss2 << (myqueries[i]+1) << " " << (myneighs[p]+1) << "\n";
                my_num_edges++;
            }

    MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);
    MPI_Reduce(&num_local_queries_made, &num_vertices, 1, MPI_INDEX, MPI_SUM, 0, comm);

    if (!myrank) ss << num_vertices << " " << num_vertices << " " << num_edges << "\n" << ss2.str();
    else std::swap(ss, ss2);

    auto sbuf = ss.str();
    std::vector<char> buf(sbuf.begin(), sbuf.end());

    MPI_Offset mysize = buf.size(), fileoffset, filesize;
    MPI_Exscan(&mysize, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    int truncate = 0;

    MPI_File fh;
    MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &filesize);
    truncate = (filesize > 0);
    MPI_Bcast(&truncate, 1, MPI_INT, 0, comm);
    if (truncate) MPI_File_set_size(fh, 0);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void DistQuery::shuffle_queues()
{
    double t = -MPI_Wtime();
    double maxtime;

    static std::random_device rd;
    static std::default_random_engine gen(rd());
    std::uniform_int_distribution<int> dist{0, nprocs-1};

    int num_trees_send = myqueue.size();
    int num_trees_recv;

    std::vector<GhostTree> sendbuf(myqueue.begin(), myqueue.end()), recvbuf;
    std::vector<GhostTreeHeader> sendbuf_headers, recvbuf_headers;
    std::vector<int> dests(num_trees_send);

    std::generate(dests.begin(), dests.end(), [&]() { return dist(gen); });

    std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);

    for (int i = 0; i < num_trees_send; ++i)
        sendcounts[dests[i]]++;

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0));
    assert((num_trees_send == sendcounts.back() + sdispls.back()));

    sendbuf_headers.resize(num_trees_send);
    auto ptrs = sdispls;

    for (int i = 0; i < num_trees_send; ++i)
    {
        sendbuf_headers[ptrs[dests[i]]++] = sendbuf[i].header;
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    num_trees_recv = recvcounts.back() + rdispls.back();
    recvbuf.resize(num_trees_recv);
    recvbuf_headers.resize(num_trees_recv);

    MPI_Datatype MPI_GHOST_TREE_HEADER;
    GhostTreeHeader::create_header_type(&MPI_GHOST_TREE_HEADER);

    MPI_Alltoallv(sendbuf_headers.data(), sendcounts.data(), sdispls.data(), MPI_GHOST_TREE_HEADER,
                  recvbuf_headers.data(), recvcounts.data(), rdispls.data(), MPI_GHOST_TREE_HEADER, comm);

    MPI_Type_free(&MPI_GHOST_TREE_HEADER);

    for (int i = 0; i < num_trees_recv; ++i)
    {
        recvbuf[i].allocate(recvbuf_headers[i], dim);
    }

    std::vector<MPI_Request> send_requests, recv_requests;

    send_requests.reserve(6*num_trees_send);
    recv_requests.reserve(6*num_trees_recv);

    for (int i = 0; i < num_trees_send; ++i) sendbuf[i].isend(dests[i],       comm, send_requests);
    for (int i = 0; i < num_trees_recv; ++i) recvbuf[i].irecv(MPI_ANY_SOURCE, comm, recv_requests);

    MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(static_cast<int>(recv_requests.size()), recv_requests.data(), MPI_STATUSES_IGNORE);

    myqueue.assign(recvbuf.begin(), recvbuf.end());

    t += MPI_Wtime();

    if (verbosity > 1)
    {
        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) { printf("[v2,time=%.3f] shuffled queues\n", maxtime); }
        fflush(stdout);
    }
}

void DistQuery::static_balancing()
{
    double t;

    t = -MPI_Wtime();
    for (auto& tree : myqueue) make_tree_queries(tree, -1);
    t += MPI_Wtime();

    if (verbosity > 1) report_finished(t);
}

void DistQuery::random_stealing(Index queries_per_tree)
{
    /*
     * 1. TIME STEAL TIME AND POLL TIME SEPARATELY
     */

    WorkStealer work_stealer(dim, comm);

    Index totsize = myqueue.size();
    MPI_Allreduce(MPI_IN_PLACE, &totsize, 1, MPI_INDEX, MPI_SUM, comm);

    double t;
    double my_comp_time = 0;
    double my_steal_time = 0;
    double my_poll_time = 0;
    double my_response_time = 0;
    double my_allreduce_time = 0;

    int num_trees_stolen = 0;
    int num_steal_reqs_recv = 0;
    int num_steal_reqs_sent = 0;
    

    int flag;
    Index mycount = 0;
    Index sendbuf = 0, recvbuf;
    MPI_Request request;

    t = -MPI_Wtime();
    MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_INDEX, MPI_SUM, comm, &request);
    t += MPI_Wtime();
    my_allreduce_time += t;

    Index iters = 0;

    while (!work_stealer.finished())
    {
        work_stealer.poll_incoming_requests(myqueue, my_poll_time, my_response_time, num_trees_stolen, num_steal_reqs_recv, num_steal_reqs_sent);

        if (!myqueue.empty())
        {
            t = -MPI_Wtime();

            if (make_tree_queries(myqueue.front(), queries_per_tree))
            {
                myqueue.pop_front();
                mycount++;
            }

            t += MPI_Wtime();
            my_comp_time += t;
        }
        else
        {
            t = -MPI_Wtime();
            work_stealer.random_steal(myqueue);
            t += MPI_Wtime();
            my_steal_time += t;
        }

        t = -MPI_Wtime();
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        t += MPI_Wtime();
        my_allreduce_time += t;

        if (flag)
        {
            t = -MPI_Wtime();
            if (recvbuf >= totsize)
                break;

            sendbuf = mycount;
            MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_INDEX, MPI_SUM, comm, &request);
            t += MPI_Wtime();
            my_allreduce_time += t;
        }
    }

    if (verbosity > 1)
    {
        printf("[v2,rank=%d,comp_time=%.3f,steal_time=%.3f,poll_time=%.3f,resp_time=%.3f,iallreduce_time=%.3f,num_trees_stolen=%d,num_steal_reqs_recv=%d,num_steal_reqs_sent=%d] completed queries [num_total_queries=%lld]\n", myrank, my_comp_time, my_steal_time, my_poll_time, my_response_time, my_allreduce_time, num_trees_stolen, num_steal_reqs_recv, num_steal_reqs_sent, num_local_queries_made); fflush(stdout);
    }

    MPI_Barrier(comm);
    fflush(stdout);
}
