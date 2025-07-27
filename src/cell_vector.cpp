#include "cell_vector.h"
#include "global_point.h"

void CellVector::reserve(Index newcap)
{
    PointVector::reserve(newcap);
    indices.reserve(newcap);
    dists.reserve(newcap);
}

void CellVector::resize(Index newsize)
{
    PointVector::resize(newsize);
    indices.resize(newsize);
    dists.resize(newsize);
}

void CellVector::clear()
{
    PointVector::clear();
    indices.clear();
    dists.clear();
}

void CellVector::push_back(const GlobalPoint& p)
{
    PointVector::push_back(p.p);
    indices.push_back(p.id);
    dists.push_back(p.dist);
}

void CellVector::sort_by_dists()
{

}
