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


void WorkStealer::poll_incoming_requests(std::deque<GhostTree>& myqueue)
{
    MPI_Status status;
    int tag, flag, source;

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
                MPI_Send(&myqueue.back().header, 1, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm);

                std::vector<MPI_Request> sendreqs;
                sendreqs.reserve(6);
                myqueue.back().isend(source, comm, sendreqs);
                MPI_Waitall(6, sendreqs.data(), MPI_STATUSES_IGNORE);
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

            int count;
            MPI_Get_count(&status, MPI_GHOST_TREE_HEADER, &count);

            GhostTreeHeader header;
            MPI_Recv(&header, count, MPI_GHOST_TREE_HEADER, source, STEAL_RESPONSE_TAG, comm, MPI_STATUS_IGNORE);

            if (count != 0)
            {
                std::vector<MPI_Request> recvreqs;
                recvreqs.reserve(6);

                myqueue.emplace_front();
                myqueue.front().allocate(header, dim);
                myqueue.front().irecv(source, comm, recvreqs);

                MPI_Waitall(6, recvreqs.data(), MPI_STATUSES_IGNORE);
            }

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
}

void WorkStealer::random_steal(std::deque<GhostTree>& myqueue)
{
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
