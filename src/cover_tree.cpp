#include "cover_tree.h"
#include <algorithm>
#include <math.h>
#include <assert.h>

void CoverTree::build(const PointVector& points, Real cover, Index leaf_size)
{
    struct Vertex
    {
        Index index;
        Real radius;
        IndexVector children;

        Vertex() {}
        Vertex(Index index, Real radius) : index(index), radius(radius) {}
        Vertex(Index index, Real radius, IndexIter first, IndexIter last) : index(index), radius(radius), children(first, last) {}
    };

    using VertexVector = std::vector<Vertex>;

    struct Hub
    {
        Index candidate, vertex, level;
        Real radius;

        IndexVector sites, ids, cells;
        RealVector dists;

        std::vector<Hub> children, leaves;

        Index size() const { return ids.size(); }

        void compute_child_hubs(const PointVector& points, Real cover, Index leaf_size, Real maxdist)
        {
            Real sep;
            Index m = size();

            Real target = std::pow(cover, (-level - 1.0)) * maxdist;

            do
            {
                Index new_candidate = candidate;
                sites.push_back(new_candidate);

                candidate = 0;
                sep = 0;

                for (Index j = 0; j < m; ++j)
                {
                    Real d = points.distance(new_candidate, ids[j]);

                    if (d < dists[j])
                    {
                        dists[j] = d;
                        cells[j] = new_candidate;
                    }

                    if (dists[j] > sep)
                    {
                        sep = dists[j];
                        candidate = ids[j];
                    }
                }

            } while (sep > target);

            for (Index site : sites)
            {
                children.emplace_back();
                Hub& child = children.back();
                Index relcand = 0;

                for (Index j = 0; j < m; ++j)
                {
                    if (cells[j] == site)
                    {
                        child.ids.push_back(ids[j]);
                        child.cells.push_back(cells[j]);
                        child.dists.push_back(dists[j]);

                        if (child.dists.back() > child.dists[relcand])
                            relcand = child.dists.size()-1;
                    }
                }

                child.sites.assign({site});
                child.candidate = child.ids[relcand];
                child.radius = child.dists[relcand];
                child.level = level+1;

                if (child.size() <= leaf_size || child.radius <= std::numeric_limits<Real>::epsilon())
                {
                    leaves.push_back(child);
                    children.pop_back();
                }
            }
        }
    };

    clear_tree();

    VertexVector vertices;

    std::deque<Hub> hubs;

    hubs.emplace_back();
    Hub& root_hub = hubs.back();

    Index n = points.num_points();

    root_hub.sites.assign({0});
    root_hub.ids.resize(n);
    root_hub.cells.resize(n, 0);
    root_hub.dists.resize(n);
    root_hub.radius = 0;
    root_hub.level = 0;

    for (Index i = 0; i < n; ++i)
    {
        root_hub.ids[i] = i;
        root_hub.dists[i] = points.distance(0,i);

        if (root_hub.dists[i] > root_hub.radius)
        {
            root_hub.radius = root_hub.dists[i];
            root_hub.candidate = i;
        }
    }

    Real maxdist = root_hub.radius;

    vertices.emplace_back(root_hub.sites.front(), root_hub.radius);
    root_hub.vertex = 0;

    while (!hubs.empty())
    {
        Hub hub = hubs.front(); hubs.pop_front();

        hub.compute_child_hubs(points, cover, leaf_size, maxdist);

        for (Hub& child : hub.children)
        {
            Index vertex = vertices.size();
            vertices.emplace_back(child.sites.front(), child.radius);
            hubs.push_back(child);
            hubs.back().vertex = vertex;
            hubs.back().level = hub.level+1;
            vertices[hub.vertex].children.push_back(vertex);
        }

        for (Hub& leaf_hub : hub.leaves)
        {
            Index vertex = vertices.size();
            vertices.emplace_back(leaf_hub.sites.front(), leaf_hub.radius);
            vertices[hub.vertex].children.push_back(vertex);

            if (leaf_hub.ids.size() > 1)
            {
                for (Index leaf : leaf_hub.ids)
                {
                    Index leaf_vertex = vertices.size();
                    vertices.emplace_back(leaf, 0);
                    vertices[vertex].children.push_back(leaf_vertex);
                }
            }
        }
    }

    Index m = vertices.size();
    this->allocate(m);

    IndexQueue queue = {0};
    IndexVector ordering = {0};
    IndexVector old_to_new(m);

    while (!queue.empty())
    {
        Index u = queue.back(); queue.pop_back();

        auto first1 = vertices[u].children.rbegin();
        auto last1 = vertices[u].children.rend();

        std::copy(first1, last1, std::back_inserter(queue));

        auto first2 = vertices[u].children.begin();
        auto last2 = vertices[u].children.end();

        for (; first2 != last2; ++first2)
        {
            ordering.push_back(*first2);
        }
    }

    assert((ordering.size() == m));

    auto it = childarr.begin();

    for (Index i = 0; i < m; ++i)
    {
        old_to_new[ordering[i]] = i;
        childptrs[i] = it - childarr.begin();
        centers[i] = vertices[ordering[i]].index;
        radii[i] = vertices[ordering[i]].radius;

        it = std::copy(vertices[ordering[i]].children.begin(), vertices[ordering[i]].children.end(), it);
    }

    childptrs[m] = childarr.size();
    std::for_each(childarr.begin(), childarr.end(), [&](Index& id) { id = old_to_new[id]; });
}

Index CoverTree::radius_query(const PointVector& points, const Atom *query, Real radius, IndexVector& neighs) const
{
    Index found = 0;
    IndexQueue queue = {0};

    while (!queue.empty())
    {
        Index u = queue.front(); queue.pop_front();

        auto first = child_begin(u);
        auto last = child_end(u);

        if (first == last)
        {
            Index leaf = centers[u];
            Real dist = points.distance(leaf, query);

            if (dist <= radius)
            {
                neighs.push_back(leaf);
                found++;
            }
        }
        else
        {
            for (; first != last; ++first)
            {
                Index child = *first;
                Real epsilon = radii[child] + radius;

                if (points.distance(centers[child], query) <= epsilon)
                    queue.push_back(child);
            }
        }
    }

    return found;
}

Index CoverTree::radius_query_indexed(const PointVector& points, const IndexVector& indices, Index query, Real radius, IndexVector& neighs) const
{
    IndexVector tmpneighs;
    Index found = radius_query(points, points[query], radius, tmpneighs);
    std::transform(tmpneighs.begin(), tmpneighs.end(), std::back_inserter(neighs), [&indices](Index id) { return indices[id]; });
    return found;
}

Index CoverTree::radius_query_indexed(const PointVector& points, const IndexVector& indices, const Atom *query, Real radius, IndexVector& neighs) const
{
    IndexVector tmpneighs;
    Index found = radius_query(points, query, radius, tmpneighs);
    std::transform(tmpneighs.begin(), tmpneighs.end(), std::back_inserter(neighs), [&indices](Index id) { return indices[id]; });
    return found;
}

void CoverTree::allocate(Index num_verts)
{
    childarr.resize(num_verts-1);
    childptrs.resize(num_verts+1);
    centers.resize(num_verts);
    radii.resize(num_verts);
}
