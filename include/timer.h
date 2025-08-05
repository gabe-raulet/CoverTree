#ifndef TIMER_H_
#define TIMER_H_

#include <mpi.h>
#include <string>

class Timer
{
    public:

        Timer(MPI_Comm comm, int root=0);
        ~Timer() { wait(); }

        void start();
        void stop();
        void wait();
        bool test();

        double get_my_time() const { return mytime; }
        double get_max_time() const { return maxtime; }

        std::string repr() const;
        std::string myrepr() const;

    private:

        double mytime;
        double maxtime;
        MPI_Request req;

        MPI_Comm comm;
        int myrank, nprocs;
        int root;
};

#endif
