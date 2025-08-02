#ifndef DIST_POINT_VECTOR_H_
#define DIST_POINT_VECTOR_H_

#include "point_vector.h"
#include <mpi.h>

class DistPointVector : public PointVector
{
    public:

        DistPointVector(MPI_Comm comm);
        DistPointVector(const PointVector& mypoints, MPI_Comm comm);

        Index getmysize() const { return mysize; }
        Index getmyoffset() const { return myoffset; }
        Index gettotsize() const { return totsize; }

        int getmyrank() const { return myrank; }
        int getnprocs() const { return nprocs; }
        MPI_Comm getcomm() const { return comm; }

        void read_fvecs(const char *fname);

    protected:

        MPI_Comm comm;
        int myrank, nprocs;
        Index mysize, myoffset, totsize;
};

#endif
