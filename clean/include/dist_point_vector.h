#ifndef DIST_POINT_VECTOR_H_
#define DIST_POINT_VECTOR_H_

#include "point_vector.h"
#include <mpi.h>
#include <algorithm>

#ifndef MAX_DIM
#error "MAX_DIM must be defined!"
#elif (MAX_DIM <= 0)
#error "MAX_DIM must be positive integer!"
#endif

class DistPointVector : public PointVector
{
    public:

        DistPointVector(const PointVector& mypoints, MPI_Comm comm);
        DistPointVector(const char *fname, MPI_Comm comm);
        ~DistPointVector();

        Index getmysize() const { return mysize; }
        Index getmyoffset() const { return myoffset; }
        Index gettotsize() const { return totsize; }

        int getmyrank() const { return myrank; }
        int getnprocs() const { return nprocs; }
        MPI_Comm getcomm() const { return comm; }

        PointVector allgather(const IndexVector& myindices, IndexVector& indices) const;
        PointVector allgather(const IndexVector& indices) const;
        PointVector gather_rma(const IndexVector& indices) const;

    protected:

        MPI_Comm comm;
        int myrank, nprocs;
        Index mysize, myoffset, totsize;
        IndexVector offsets;

        MPI_Win win;
        MPI_Datatype MPI_POINT;

        int point_owner(Index id) const { return (std::upper_bound(offsets.begin(), offsets.end(), id) - offsets.begin())-1; }

    private:

        void init_comm();
        void init_offsets();
        void init_window();
};

#endif
