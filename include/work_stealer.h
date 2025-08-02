#ifndef WORK_STEALER_H_
#define WORK_STEALER_H_

#include "ghost_tree.h"
#include <random>
#include <mpi.h>

#define STEAL_REQUEST_TAG 0
#define STEAL_RESPONSE_TAG 1
#define TOKEN_TAG 2
#define SHUTDOWN_TAG 3

#define WHITE_TOKEN 4
#define BLACK_TOKEN 5
#define NULL_TOKEN 6

static std::random_device rd;
static std::default_random_engine gen(rd());

class WorkStealer
{
    public:

        WorkStealer(int dim, MPI_Comm comm);
        ~WorkStealer();

        bool finished();
        void poll_incoming_requests(std::deque<GhostTree>& myqueue, double& my_poll_time, double& my_response_time);
        void random_steal(std::deque<GhostTree>& myqueue);
        void poll_global_termination();

    private:

        MPI_Comm comm;
        int myrank, nprocs;

        int token;
        int color;
        int dest;
        bool done;
        bool first;
        bool steal_in_progress;
        MPI_Request request;
        MPI_Datatype MPI_GHOST_TREE_HEADER;

        int dim; /* this pesky dim! */
};

#endif
