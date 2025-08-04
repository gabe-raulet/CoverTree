#include "timer.h"
#include <stdio.h>

Timer::Timer(MPI_Comm comm, int root) : comm(comm), root(root)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
}

void Timer::start()
{
    mytime = -MPI_Wtime();
}

void Timer::stop()
{
    mytime += MPI_Wtime();

    MPI_Request reqs[2];
    MPI_Ireduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, root, comm, &reqs[0]);
    MPI_Ireduce(&mytime, &avgtime, 1, MPI_DOUBLE, MPI_SUM, root, comm, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    avgtime /= nprocs;
}

std::string Timer::repr() const
{
    char buf[512];
    snprintf(buf, 512, "time=%.3f", maxtime);
    return std::string(buf);
}

std::string Timer::repr_long() const
{
    char buf[512];
    snprintf(buf, 512, "maxtime=%.3f,avgtime=%.3f", maxtime, avgtime);
    return std::string(buf);
}
