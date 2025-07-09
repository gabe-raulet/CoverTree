template <class Metric>
void CoverTree<Metric>::build(Real cover, Index leaf_size)
{
    struct Hub
    {
        Index candidate, vertex, level;
        Real radius;

        IndexVector sites, ids, cells;
        RealVector dists;

        std::vector<Hub> children, leaves;

        Index size() const { return ids.size(); }

        void compute_child_hubs(const Metric& metric, Real cover, Index leaf_size, Real maxdist)
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
                    Real d = metric.distance(new_candidate, ids[j]);

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

    vertices.clear();

    std::deque<Hub> hubs;

    hubs.emplace_back();
    Hub& root_hub = hubs.back();

    Index n = num_points();

    root_hub.sites.assign({0});
    root_hub.ids.resize(n);
    root_hub.cells.resize(n, 0);
    root_hub.dists.resize(n);
    root_hub.radius = 0;
    root_hub.level = 0;

    for (Index i = 0; i < n; ++i)
    {
        root_hub.ids[i] = i;
        root_hub.dists[i] = metric.distance(0, i);

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

        hub.compute_child_hubs(metric, cover, leaf_size, maxdist);

        for (Hub& child : hub.children)
        {
            Index vertex = vertices.size();
            vertices.emplace_back(child.sites.front(), child.radius);
            hubs.push_back(child);
            hubs.back().vertex = vertex;
            hubs.back().level = hub.level+1;
            vertices.back().level = hub.level+1;
            vertices[hub.vertex].children.push_back(vertex);
        }

        for (Hub& leaf_hub : hub.leaves)
        {
            Index vertex = vertices.size();
            vertices.emplace_back(leaf_hub.sites.front(), leaf_hub.radius);
            vertices.back().level = leaf_hub.level;

            vertices[hub.vertex].children.push_back(vertex);

            for (Index leaf : leaf_hub.ids)
            {
                vertices[vertex].leaves.push_back(leaf);
            }
        }
    }

    maxlevel = std::ceil(std::log(maxdist)/std::log(cover));
    numverts = vertices.size();
}

template <class Metric>
typename CoverTreeInterface<Metric>::Index
CoverTreeInterface<Metric>::radius_query(const Atom *query, Real radius, RealVector& dists, IndexVector& neighs) const
{
    Index found = 0;
    std::deque<Index> queue = {0};

    while (!queue.empty())
    {
        Index u = queue.front(); queue.pop_front();

        Index nleaf, nchild;
        const Index *leaves = get_leaves(u, nleaf);
        const Index *children = get_children(u, nchild);

        for (Index i = 0; i < nleaf; ++i)
        {
            Index leaf = leaves[i];
            Real d = metric.distance(leaf, query);

            if (d <= radius)
            {
                neighs.push_back(leaf);
                dists.push_back(d);
                found++;
            }
        }

        for (Index i = 0; i < nchild; ++i)
        {
            Index v = children[i];
            Index idx = get_index(v);
            Real vradius = get_radius(v);

            if (metric.distance(idx, query) <= vradius + radius)
                queue.push_back(v);
        }
    }

    return found;
}

template <class Metric>
PackedCoverTree<Metric>::PackedCoverTree(const CoverTree<Metric>& tree) : Base(tree.get_metric())
{
    maxlevel = tree.max_level();
    numverts = tree.num_vertices();

    vertices.resize(tree.num_vertices());
    ids.resize(tree.num_points() + tree.num_vertices());

    auto it = ids.begin();

    for (Index v = 0; v < numverts; ++v)
    {
        Index nleaf, nchild;
        const Index *leaves = tree.get_leaves(v, nleaf);
        const Index *children = tree.get_children(v, nchild);

        vertices[v].index = tree.get_index(v);
        vertices[v].level = tree.get_level(v);
        vertices[v].radius = tree.get_radius(v);
        vertices[v].nchild = nchild;
        vertices[v].nleaf = nleaf;

        vertices[v].childptr = it - ids.begin();
        it = std::copy(children, children+nchild, it);

        vertices[v].leafptr = it - ids.begin();
        it = std::copy(leaves, leaves+nleaf, it);
    }
}
