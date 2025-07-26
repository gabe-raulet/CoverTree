#include "point_vector.h"
#include <filesystem>

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

std::string PointVector::repr() const
{
    char buf[512];
    snprintf(buf, 512, "PointVector(size=%lld,dim=%d)", size, dim);
    return std::string(buf);
}
