#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "point_vector.h"

class CoverTree
{
    public:

        CoverTree() = delete;
        CoverTree(const PointVector& points) : points(points) {}

        void build(Real cover, Index leaf_size);

        Index num_vertices() const { return centers.size(); }
        Index num_points() const { return points.num_points(); }

        Index radius_query(const Atom *query, Real radius, IndexVector& neighs) const;
        Index radius_query(Index query, Real radius, IndexVector& neighs) const { return radius_query(points[query], radius, neighs); }

        std::string repr() const;

    private:

        PointVector points;

        IndexVector centers; /* size m; vertex centers */
        IndexVector childarr; /* size m-1; children array */
        IndexVector childptrs; /* size m+1; children poitners */
        RealVector radii; /* size m; vertex radii */

        IndexIter child_begin(Index vertex) const { return childarr.begin() + childptrs[vertex]; }
        IndexIter child_end(Index vertex) const { return childarr.begin() + childptrs[vertex+1]; }

        void clear_tree() { centers.clear(); childarr.clear(); childptrs.clear(); radii.clear(); }
};

#endif
