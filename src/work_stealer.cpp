#include "work_stealer.h"

WorkStealer::WorkStealer(int dim, MPI_Comm comm)
    : comm(comm),
      token(NULL_TOKEN),
      color(WHITE_TOKEN),
      done(false),
      first(true),
      steal_in_progress(false),
      request(MPI_REQUEST_NULL),
      dim(dim)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    /*
     * Root rank starts out with the token,
     * set to the color white.
     */
    if (!myrank)
        token = WHITE_TOKEN;

    dest = (myrank+1)%nprocs;

    GhostTreeHeader::create_header_type(&MPI_GHOST_TREE_HEADER);
}

WorkStealer::~WorkStealer()
{
    MPI_Type_free(&MPI_GHOST_TREE_HEADER);
}

bool WorkStealer::finished()
{
    return done;
}


void WorkStealer::poll_incoming_requests(std::deque<GhostTree>& myqueue, double& my_poll_time, double& my_response_time)
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

            if (myqueue.size() <= 1)
            {
                MPI_Send(MPI_BOTTOM, 0, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm);
            }
            else
            {

                int queue_size = myqueue.size();
                int num_trees_send = std::max(1, queue_size/3);

                std::vector<MPI_Request> sendreqs;

                std::vector<GhostTreeHeader> headers(num_trees_send);

                for (int i = 0; i < num_trees_send; ++i)
                    headers[i] = myqueue[queue_size-i-1].header;

                MPI_Send(headers.data(), num_trees_send, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm);

                sendreqs.reserve(6*num_trees_send);

                for (int i = 0; i < num_trees_send; ++i)
                    myqueue[queue_size-i-1].isend(source, comm, sendreqs);

                MPI_Waitall(6*num_trees_send, sendreqs.data(), MPI_STATUSES_IGNORE);

                for (int i = 0; i < num_trees_send; ++i)
                    myqueue.pop_back();

                if (source < myrank)
                {
                    color = BLACK_TOKEN;
                }
            }
        }

        if (tag == STEAL_RESPONSE_TAG)
        {
            MPI_Wait(&request, MPI_STATUS_IGNORE);

            /* TIME START HERE (This is where actual communication is happening) */
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
                    myqueue.emplace_front();

                for (int i = 0; i < num_trees_recv; ++i)
                    myqueue[i].allocate(headers[i], dim);

                for (int i = 0; i < num_trees_recv; ++i)
                    myqueue[i].irecv(source, comm, recvreqs);


                MPI_Waitall(6*num_trees_recv, recvreqs.data(), MPI_STATUSES_IGNORE);
            }

            /* TIME END */

            t += MPI_Wtime();
            response_time += t;

            steal_in_progress = false;
        }

        if (tag == TOKEN_TAG)
        {
            MPI_Recv(&token, 1, MPI_INT, source, TOKEN_TAG, comm, MPI_STATUS_IGNORE);
        }

        if (tag == SHUTDOWN_TAG)
        {
            MPI_Recv(MPI_BOTTOM, 0, MPI_BYTE, source, SHUTDOWN_TAG, comm, MPI_STATUS_IGNORE);
            done = true;
        }
    }

    tottime += MPI_Wtime();
    my_poll_time += (tottime - response_time);
    my_response_time += response_time;
}

void WorkStealer::random_steal(std::deque<GhostTree>& myqueue)
{
    if (steal_in_progress)
        return;

    std::uniform_int_distribution<int> dist{0,nprocs-1};

    int victim;

    do { victim = dist(gen); } while (victim == myrank);

    MPI_Isend(MPI_BOTTOM, 0, MPI_BYTE, victim, STEAL_REQUEST_TAG, comm, &request);

    steal_in_progress = true;
}

void WorkStealer::poll_global_termination()
{
    if (nprocs == 1)
    {
        done = true;
    }
    else
    {
        if (!myrank && token == WHITE_TOKEN && color == WHITE_TOKEN)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                std::vector<MPI_Request> reqs(nprocs-1);

                for (int i = 1; i < nprocs; ++i)
                {
                    MPI_Isend(MPI_BOTTOM, 0, MPI_BYTE, i, SHUTDOWN_TAG, comm, &reqs[i-1]);
                }

                MPI_Waitall(nprocs-1, reqs.data(), MPI_STATUSES_IGNORE);
                done = true;
            }
        }

        if (token != NULL_TOKEN)
        {
            if (!myrank)
            {
                color = token = WHITE_TOKEN;
                MPI_Send(&token, 1, MPI_INT, dest, TOKEN_TAG, comm);
                token = NULL_TOKEN;

            }
            else
            {
                if (color == WHITE_TOKEN)
                {
                    MPI_Send(&token, 1, MPI_INT, dest, TOKEN_TAG, comm);
                    token = NULL_TOKEN;
                }
                else if (color == BLACK_TOKEN)
                {
                    token = BLACK_TOKEN;
                    color = WHITE_TOKEN;
                    MPI_Send(&token, 1, MPI_INT, dest, TOKEN_TAG, comm);
                    token = NULL_TOKEN;
                }
            }
        }
    }
}
