#include "global_termination.h"

GlobalTermination::GlobalTermination(Index global_count, MPI_Comm comm)
    : flag(0),
      global_count(global_count),
      local_count(0),
      comm(comm) { restart_allreduce(); }

void GlobalTermination::restart_allreduce()
{
    sendbuf = local_count;
    MPI_Iallreduce(&sendbuf, &recvbuf, 1, MPI_INDEX, MPI_SUM, comm, &request);
}

bool GlobalTermination::done()
{
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);

    if (flag)
    {
        if (recvbuf >= global_count)
        {
            return true;
        }
        else
        {
            restart_allreduce();
            return false;
        }
    }
    else
    {
        return false;
    }
}

void GlobalTermination::increment()
{
    local_count++;
}

