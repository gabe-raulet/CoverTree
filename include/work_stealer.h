#ifndef WORK_STEALER_H_
#define WORK_STEALER_H_

#include "ghost_tree.h"
#include <random>
#include <mpi.h>

#define STEAL_REQUEST_TAG 0
#define STEAL_RESPONSE_TAG 1

static std::random_device rd;
static std::default_random_engine gen(rd());

class WorkStealer
{
    public:

        WorkStealer(std::deque<GhostTree> *myqueue, int dim, MPI_Comm comm);
        ~WorkStealer();

        bool finished();
        void poll_incoming_requests(double& my_poll_time, double& my_response_time);
        void random_steal();

        Index steal_attempts;
        Index steal_successes;
        Index steal_services;
        Index queries_remaining;

    private:

        MPI_Comm comm;
        int myrank, nprocs;

        bool steal_in_progress;
        MPI_Request request;
        MPI_Datatype MPI_GHOST_TREE_HEADER;

        int dim; /* this pesky dim! */

        std::deque<GhostTree> *myqueue;
};

#endif
