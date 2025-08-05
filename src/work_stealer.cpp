#include "work_stealer.h"
#include <algorithm>

WorkStealer::WorkStealer(std::deque<GhostTree> *myqueue, int dim, MPI_Comm comm)
    : comm(comm),
      steal_in_progress(false),
      request(MPI_REQUEST_NULL),
      steal_attempts(0), steal_successes(0), steal_services(0),
      queries_remaining(0),
      dim(dim),
      myqueue(myqueue)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    GhostTreeHeader::create_header_type(&MPI_GHOST_TREE_HEADER);

    for (auto it = myqueue->begin(); it != myqueue->end(); ++it)
    {
        queries_remaining += it->header.queries_remaining();
    }
}

WorkStealer::~WorkStealer()
{
    myqueue = nullptr;
    MPI_Type_free(&MPI_GHOST_TREE_HEADER);
}

void WorkStealer::poll_incoming_requests(double& my_poll_time, double& my_response_time)
{
    /* SORT THE QUEUES BY WORK ESTIMATE */

    MPI_Status status;
    int tag, flag, source;

    double response_time = 0;
    double tottime = -MPI_Wtime();
    double t;

    while (true)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
        if (!flag) break;

        tag = status.MPI_TAG;
        source = status.MPI_SOURCE;

        if (tag == STEAL_REQUEST_TAG)
        {
            MPI_Recv(MPI_BOTTOM, 0, MPI_BYTE, source, STEAL_REQUEST_TAG, comm, MPI_STATUS_IGNORE);

            if (myqueue->size() <= 1)
            {
                MPI_Send(MPI_BOTTOM, 0, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm);
            }
            else
            {
                std::sort(myqueue->begin(), myqueue->end(), [](const auto& a, const auto& b) { return a.tree.num_vertices() > b.tree.num_vertices(); });

                int queue_size = myqueue->size();
                int num_trees_send = 1;
                /* int num_trees_send = std::max(1, queue_size/3); */

                std::vector<MPI_Request> sendreqs;

                std::vector<GhostTreeHeader> headers(num_trees_send);

                for (int i = 0; i < num_trees_send; ++i)
                {
                    headers[i] = (*myqueue)[queue_size-i-1].header;
                    queries_remaining -= headers[i].queries_remaining();
                }

                MPI_Send(headers.data(), num_trees_send, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm);

                sendreqs.reserve(6*num_trees_send);

                for (int i = 0; i < num_trees_send; ++i)
                    (*myqueue)[queue_size-i-1].isend(source, comm, sendreqs);

                MPI_Waitall(6*num_trees_send, sendreqs.data(), MPI_STATUSES_IGNORE);

                for (int i = 0; i < num_trees_send; ++i)
                    myqueue->pop_back();

                steal_services++;
            }
        }

        if (tag == STEAL_RESPONSE_TAG)
        {
            MPI_Wait(&request, MPI_STATUS_IGNORE);

            t = -MPI_Wtime();
            int num_trees_recv;
            MPI_Get_count(&status, MPI_GHOST_TREE_HEADER, &num_trees_recv);

            std::vector<GhostTreeHeader> headers(num_trees_recv);
            MPI_Recv(headers.data(), num_trees_recv, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm, MPI_STATUS_IGNORE);

            if (num_trees_recv != 0)
            {
                std::vector<MPI_Request> recvreqs;
                recvreqs.reserve(6*num_trees_recv);

                for (int i = 0; i < num_trees_recv; ++i)
                    myqueue->emplace_front();

                for (int i = 0; i < num_trees_recv; ++i)
                {
                    (*myqueue)[i].allocate(headers[i], dim);
                    queries_remaining += headers[i].queries_remaining();
                }

                for (int i = 0; i < num_trees_recv; ++i)
                    (*myqueue)[i].irecv(source, comm, recvreqs);

                MPI_Waitall(6*num_trees_recv, recvreqs.data(), MPI_STATUSES_IGNORE);

                steal_successes++;
            }

            t += MPI_Wtime();
            response_time += t;

            steal_in_progress = false;
        }
    }

    tottime += MPI_Wtime();
    my_poll_time += (tottime - response_time);
    my_response_time += response_time;
}

void WorkStealer::random_steal()
{
    if (steal_in_progress)
        return;

    std::uniform_int_distribution<int> dist{0,nprocs-1};

    int victim;

    do { victim = dist(gen); } while (victim == myrank);

    MPI_Isend(MPI_BOTTOM, 0, MPI_BYTE, victim, STEAL_REQUEST_TAG, comm, &request);

    steal_in_progress = true;
    steal_attempts++;
}
