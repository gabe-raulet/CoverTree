#include "dist_point_vector.h"
#include <assert.h>
#include <filesystem>

DistPointVector::DistPointVector(MPI_Comm comm)
    : comm(comm),
      mysize(0), myoffset(0), totsize(0)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
}

DistPointVector::DistPointVector(const PointVector& mypoints, MPI_Comm comm)
    : PointVector(mypoints),
      comm(comm),
      mysize(mypoints.num_points()), myoffset(0), totsize(0)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    MPI_Allreduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, comm);
}

void DistPointVector::read_fvecs(const char *fname)
{
    int d;
    FILE *f;
    Index filesize, myleft;
    std::filesystem::path path;

    if (!myrank)
    {
        path = fname;
        filesize = std::filesystem::file_size(path);

        f = fopen(fname, "rb");
        fread(&d, sizeof(int), 1, f);
        fclose(f);
    }

    MPI_Request reqs[2];
    MPI_Ibcast(&d, 1, MPI_INT, 0, comm, reqs);
    MPI_Ibcast(&filesize, 1, MPI_INDEX, 0, comm, reqs+1);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    size_t disp = 4 * (d + 1);

    assert((sizeof(Atom) == 4));
    assert((filesize % disp == 0));

    totsize = filesize / disp;

    mysize = totsize / nprocs;
    myleft = totsize % nprocs;

    if (myrank < myleft)
        mysize++;

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    MPI_Offset fileoffset = myoffset*disp;
    std::vector<char> mybuf(mysize*disp);

    MPI_File fh;
    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at_all(fh, fileoffset, mybuf.data(), static_cast<int>(mybuf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    PointVector::dim = d;

    clear();
    resize(mysize);

    char *ptr = mybuf.data();
    auto it = PointVector::atoms.begin();

    for (Index i = 0; i < mysize; ++i)
    {
        Atom *pt = (Atom *)(ptr + sizeof(int));
        it = std::copy(pt, pt+dim, it);
        ptr += disp;
    }
}
