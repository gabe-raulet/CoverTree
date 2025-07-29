#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "point_vector.h"
#include <assert.h>

class CoverTree
{
    public:

        CoverTree() = default;

        void build(const PointVector& points, Real cover, Index leaf_size);

        Index num_vertices() const { assert((buf.size() % 3 == 0)); return buf.size()/3; } /* hacky! address later */

        Index radius_query(const PointVector& points, const Atom *query, Real radius, IndexVector& neighs) const;
        Index radius_query(const PointVector& points, Index query, Real radius, IndexVector& neighs) const { return radius_query(points, points[query], radius, neighs); }
        Index radius_query_indexed(const PointVector& points, Index query, Real radius, IndexVector& neighs, const IndexVector& indices) const;

        std::string repr() const;

        void allocate(Index num_verts);

    private:

        IndexVector buf; /* totsize 3*m */
        RealVector radii; /* size m; vertex radii */

        Index *childarr; /* size m-1; children array */
        Index *childptrs; /* size m+1; children pointers */
        Index *centers; /* size m; vertex centers */

        IndexIter child_begin(Index vertex) const { return buf.begin() + childptrs[vertex]; } /* assumes childarr goes first */
        IndexIter child_end(Index vertex) const { return buf.begin() + childptrs[vertex+1]; } /* assumes childarr goes first */

        void clear_tree() { buf.clear(); radii.clear(); }
};

#endif
