#include "point_vector.h"
#include <filesystem>

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
