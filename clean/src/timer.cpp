#include "timer.h"
#include <stdio.h>
#include <assert.h>

Timer::Timer(MPI_Comm comm, int root) : req(MPI_REQUEST_NULL), comm(comm), root(root)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
}

void Timer::wait()
{
    if (req != MPI_REQUEST_NULL)
    {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        req = MPI_REQUEST_NULL;
    }
}

bool Timer::test()
{
    if (req == MPI_REQUEST_NULL)
    {
        return true;
    }
    else
    {
        int flag;
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        return (flag != 0);
    }
}

void Timer::start()
{
    wait();
    mytime = -MPI_Wtime();
}

void Timer::stop()
{
    mytime += MPI_Wtime();
    MPI_Ireduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, root, comm, &req);
}

std::string Timer::repr() const
{
    char buf[512];
    snprintf(buf, 512, "time=%.3f", maxtime);
    return std::string(buf);
}

std::string Timer::myrepr() const
{
    char buf[512];
    snprintf(buf, 512, "rank=%d,time=%.3f", myrank, mytime);
    return std::string(buf);
}
