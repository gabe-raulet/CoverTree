#include "point_vector.h"
#include <filesystem>
#include <math.h>
#include <assert.h>

Real PointVector::distance(const Atom *p, const Atom *q) const
{
    Real val = 0;
    Real delta;

    for (int i = 0; i < dim; ++i)
    {
        delta = static_cast<Real>(p[i] - q[i]);
        val += delta*delta;
    }

    return std::sqrt(val);
}

void PointVector::read_fvecs(const char *fname)
{
    Index filesize, n;
    int d;
    FILE *f;
    std::filesystem::path path;

    path = fname;
    filesize = std::filesystem::file_size(path);

    f = fopen(fname, "rb");
    fread(&d, sizeof(int), 1, f);
    fseek(f, SEEK_SET, 0);
    n = filesize / (4*(d+1));

    size = n;
    dim = d;
    atoms.resize(size*dim);
    Atom *ptr = atoms.data();

    for (Index i = 0; i < size; ++i)
    {
        fread(&d, sizeof(int), 1, f);
        fread(ptr, sizeof(Atom), dim, f);
        ptr += dim;
    }

    fclose(f);
}

void PointVector::read_fvecs(const char *fname, MPI_Comm comm)
{
    int d;
    FILE *f;
    Index mysize, myoffset, totsize, filesize, myleft;
    std::filesystem::path path;

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

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

    dim = d;

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

void PointVector::write_fvecs(const char *fname) const
{
    FILE *f = fopen(fname, "wb");

    const Atom *ptr = atoms.data();

    for (Index i = 0; i < size; ++i)
    {
        fwrite(&dim, sizeof(int), 1, f);
        fwrite(ptr, sizeof(Atom), dim, f);
        ptr += dim;
    }

    fclose(f);
}

PointVector PointVector::gather(const IndexVector& offsets) const
{
    Index newsize = offsets.size();
    AtomVector newatoms(newsize*dim);
    auto it = newatoms.begin();

    for (Index offset : offsets)
    {
        it = std::copy(begin(offset), end(offset), it);
    }

    return PointVector(newatoms.data(), newsize, dim);
}

std::string PointVector::repr() const
{
    char buf[512];
    snprintf(buf, 512, "PointVector(size=%lld,dim=%d)", size, dim);
    return std::string(buf);
}
