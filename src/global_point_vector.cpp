#include "global_point_vector.h"
#include <assert.h>
#include <numeric>
#include <filesystem>

GlobalPointVector::GlobalPointVector(const PointVector& mypoints, Index cell_init, Real dist_init, MPI_Comm comm)
    : PointVector(mypoints),
      ids(mypoints.num_points()),
      cells(mypoints.num_points(), cell_init),
      dists(mypoints.num_points(), dist_init)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);

    Index mysize = mypoints.num_points();
    Index myoffset;

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    std::iota(ids.begin(), ids.end(), myoffset);
}

void GlobalPointVector::create_mpi_type(MPI_Datatype *MPI_GLOBAL_POINT) const
{
    assert((dim <= MAX_DIM));

    int blklens[4] = {dim,1,1,1};
    MPI_Aint disps[4] = {offsetof(GlobalPoint,p), offsetof(GlobalPoint,id), offsetof(GlobalPoint,cell), offsetof(GlobalPoint,dist)};
    MPI_Datatype types[4] = {MPI_ATOM, MPI_INDEX, MPI_INDEX, MPI_REAL};
    MPI_Type_create_struct(4, blklens, disps, types, MPI_GLOBAL_POINT);
    MPI_Type_commit(MPI_GLOBAL_POINT);
}

GlobalPoint GlobalPointVector::operator[](Index offset) const
{
    return GlobalPoint(PointVector::operator[](offset), num_dimensions(), ids[offset], cells[offset], dists[offset]);
}

void GlobalPointVector::reserve(Index newcap)
{
    PointVector::reserve(newcap);
    ids.reserve(newcap);
    cells.reserve(newcap);
    dists.reserve(newcap);
}

void GlobalPointVector::resize(Index newsize)
{
    PointVector::resize(newsize);
    ids.resize(newsize);
    cells.resize(newsize);
    dists.resize(newsize);
}

void GlobalPointVector::clear()
{
    PointVector::clear();
    ids.clear();
    cells.clear();
    dists.clear();
}

void GlobalPointVector::push_back(const GlobalPoint& pt)
{
    PointVector::push_back(pt.p);
    ids.push_back(pt.id);
    cells.push_back(pt.cell);
    dists.push_back(pt.dist);
}

void GlobalPointVector::set(Index offset, const GlobalPoint& pt)
{
    PointVector::set(offset, pt.p);
    ids[offset] = pt.id;
    cells[offset] = pt.cell;
    dists[offset] = pt.dist;
}

void GlobalPointVector::read_fvecs(const char *fname, MPI_Comm comm)
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

        ids[i] = i+myoffset;
        cells[i] = -1;
        dists[i] = 0;
    }
}

std::string GlobalPointVector::repr() const
{
    char buf[512];
    snprintf(buf, 512, "GlobalPointVector(mysize=%lld,dim=%d,firstid=%lld,lastid=%lld)", num_points(), num_dimensions(), ids.front(), ids.back());
    return std::string(buf);
}
