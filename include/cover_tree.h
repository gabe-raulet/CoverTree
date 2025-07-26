#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "point_vector.h"

class CoverTree
{
    public:

        CoverTree() = delete;
        CoverTree(const PointVector& points);

        void build(Real cover, Index leaf_size);

        Index num_vertices() const;
        Index num_points() const;

        Index radius_query(const Atom *query, Real radius, IndexVector& neighs) const;
        Index radius_query(Index query, Real radius, IndexVector& neighs) const;

        std::string repr() const;

    private:

        PointVector points;

        IndexVector centers; /* size m; vertex centers */
        IndexVector childarr; /* size m-1; children array */
        IndexVector childptrs; /* size m+1; children poitners */
        RealVector radii; /* size m; vertex radii */

        IndexIter child_begin(Index vertex) const;
        IndexIter child_end(Index vertex) const;

        void clear_tree();
};

#endif
