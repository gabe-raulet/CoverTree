#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "point_vector.h"

class CoverTree
{
    public:

        CoverTree() = default;

        void build(const PointVector& points, Real cover, Index leaf_size);

        Index num_vertices() const { return centers.size(); }

        Index radius_query(const PointVector& points, const Atom *query, Real radius, IndexVector& neighs) const;
        Index radius_query(const PointVector& points, Index query, Real radius, IndexVector& neighs) const { return radius_query(points, points[query], radius, neighs); }
        Index radius_query_indexed(const PointVector& points, Index query, Real radius, IndexVector& neighs, const IndexVector& indices) const;

        std::string repr() const;

        void allocate(Index num_verts);

    private:

        IndexVector centers; /* size m; vertex centers */
        IndexVector childarr; /* size m-1; children array */
        IndexVector childptrs; /* size m+1; children poitners */
        RealVector radii; /* size m; vertex radii */

        IndexIter child_begin(Index vertex) const { return childarr.begin() + childptrs[vertex]; }
        IndexIter child_end(Index vertex) const { return childarr.begin() + childptrs[vertex+1]; }

        void clear_tree() { centers.clear(); childarr.clear(); childptrs.clear(); radii.clear(); }
};

#endif
