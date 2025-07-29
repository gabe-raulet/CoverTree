#ifndef GLOBAL_TERMINATION_H_
#define GLOBAL_TERMINATION_H_

#include "utils.h"
#include <mpi.h>

class GlobalTermination
{
    public:

        GlobalTermination(Index global_count, MPI_Comm comm);

        bool done();
        void increment();

        bool ready() const { return static_cast<bool>(flag); }

    private:

        void restart_allreduce();

        int flag;
        MPI_Request request;
        Index global_count, local_count;
        Index sendbuf, recvbuf;
        MPI_Comm comm;
};

#endif
