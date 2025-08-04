#ifndef TIMER_H_
#define TIMER_H_

#include <mpi.h>
#include <string>

class Timer
{
    public:

        Timer(MPI_Comm comm, int root=0);

        void start();
        void stop();

        double get_my_time() const { return mytime; }
        double get_max_time() const { return maxtime; }
        double get_avg_time() const { return avgtime; }

        std::string repr() const;
        std::string repr_long() const;

    private:

        double mytime;
        double maxtime;
        double avgtime;

        MPI_Comm comm;
        int myrank, nprocs;
        int root;
};

#endif
