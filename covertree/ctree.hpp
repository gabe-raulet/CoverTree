template <class Distance, class Real, class Atom>
void CoverTree<Distance, Real, Atom>::build(const Atom *points, Real cover, Index leaf_size, int num_threads)
{
    struct Hub
    {
        Index candidate, vertex, level;
        Real radius;

        IndexVector sites, ids, cells;
        RealVector dists;

        std::vector<Hub> children, leaves;

        Index size() const { return ids.size(); }

        void compute_child_hubs(const Atom *points, Index dim, Real cover, Index leaf_size, Real maxdist)
        {
            Real sep;
            Index m = size();

            Real target = std::pow(cover, (-level - 1.0)) * maxdist;

            do
            {
                Index new_candidate = candidate;
                const Atom *new_candidate_point = &points[new_candidate*dim];
                sites.push_back(new_candidate);

                candidate = 0;
                sep = 0;

                for (Index j = 0; j < m; ++j)
                {
                    Real d = distance(new_candidate_point, &points[ids[j]*dim], dim);

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

    root_hub.sites.assign({0});
    root_hub.ids.resize(n);
    root_hub.cells.resize(n, 0);
    root_hub.dists.resize(n);
    root_hub.radius = 0;
    root_hub.level = 0;

    for (Index i = 0; i < n; ++i)
    {
        root_hub.ids[i] = i;
        root_hub.dists[i] = distance(&points[0], &points[i*d], d);

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

        hub.compute_child_hubs(points, d, cover, leaf_size, maxdist);

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
}

template <class Distance, class Real, class Atom>
Index CoverTree<Distance, Real, Atom>::knn_query(const Atom *points, const Atom *query, Index k, Index *neighbors, Real *dists) const
{
    using QueueItem = std::tuple<Real, Real, Index>;
    using NeighborItem = std::tuple<Real, Index>;

    auto queue_cmp = [](const QueueItem& lhs, const QueueItem& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); };
    auto neighbor_cmp = [](const NeighborItem& lhs, const NeighborItem& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); };

    std::vector<QueueItem> queue;
    std::vector<NeighborItem> neighbor_queue;

    Real distance_upper_bound = std::numeric_limits<Real>::max();
    Real dist_to_ctr = distance(&points[0*d], query, d);
    Real min_distance = std::max(Real(0), dist_to_ctr - vertices[0].radius);

    queue.emplace_back(min_distance, dist_to_ctr, 0);
    std::make_heap(queue.begin(), queue.end(), queue_cmp);

    Real dist;

    while (!queue.empty())
    {
        auto [min_distance, dist_to_ctr, node] = queue.front(); std::pop_heap(queue.begin(), queue.end(), queue_cmp); queue.pop_back();
        Index node_point = vertices[node].index;

        for (Index child : vertices[node].children)
        {
            Index child_point = vertices[child].index;

            if (child_point == node_point) dist = dist_to_ctr;
            else dist = distance(query, &points[child_point*d], d);

            min_distance = std::max(Real(0), dist - vertices[child].radius);

            if (min_distance <= distance_upper_bound)
            {
                queue.emplace_back(min_distance, dist, child);
                std::push_heap(queue.begin(), queue.end(), queue_cmp);
            }
        }

        for (Index leaf : vertices[node].leaves)
        {
            if (leaf == node_point) dist = dist_to_ctr;
            else dist = distance(query, &points[leaf*d], d);

            if (dist <= distance_upper_bound)
            {
                if (neighbor_queue.size() == k)
                {
                    std::pop_heap(neighbor_queue.begin(), neighbor_queue.end(), neighbor_cmp);
                    neighbor_queue.pop_back();
                }

                neighbor_queue.emplace_back(-dist, leaf);
                std::push_heap(neighbor_queue.begin(), neighbor_queue.end(), neighbor_cmp);

                if (neighbor_queue.size() == k)
                {
                    distance_upper_bound = -std::get<0>(neighbor_queue.front());
                }
            }
        }
    }

    std::for_each(neighbor_queue.begin(), neighbor_queue.end(), [](auto& item) { std::get<0>(item) *= -1; });
    std::sort(neighbor_queue.begin(), neighbor_queue.end());

    Index found = 0;

    for (const auto& [dist, neigh] : neighbor_queue)
    {
        dists[found] = dist;
        neighbors[found++] = neigh;
    }

    return found;
}

template <class Distance, class Real, class Atom>
Index CoverTree<Distance, Real, Atom>::radius_query(const Atom *points, const Atom *query, Real radius, IndexVector& neighbors, RealVector& dists) const
{
    Index neighs = 0;
    std::deque<Index> queue = {0};

    while (!queue.empty())
    {
        Index u = queue.front(); queue.pop_front();
        const Vertex& u_vtx = vertices[u];

        for (Index leaf : u_vtx.leaves)
        {
            Real dist = distance(query, &points[leaf*d], d);

            if (dist <= radius)
            {
                neighbors.push_back(leaf);
                dists.push_back(dist);
                neighs++;
            }
        }

        for (Index v : u_vtx.children)
        {
            const Vertex& v_vtx = vertices[v];

            if (distance(query, &points[v_vtx.index*d], d) <= v_vtx.radius + radius)
                queue.push_back(v);
        }
    }

    return neighs;
}

template <class Distance, class Real, class Atom>
Index CoverTree<Distance, Real, Atom>::radius_neighbors_graph(const Atom *points, Real radius, std::vector<IndexVector>& neighbors, std::vector<RealVector>& dists, int num_threads) const
{
    Index nz = 0;
    neighbors.clear(), dists.clear();
    neighbors.resize(n), dists.resize(n);

    omp_set_num_threads(num_threads);

    #pragma omp parallel for reduction(+:nz)
    for (Index i = 0; i < n; ++i)
    {
        nz += radius_query(points, &points[i*d], radius, neighbors[i], dists[i]);
    }

    return nz;
}

template <class Distance, class Real, class Atom>
Index CoverTree<Distance, Real, Atom>::kneighbors_graph(const Atom *points, Index k, IndexVector& neighbors, RealVector& dists, IndexVector& found, int num_threads) const
{
    omp_set_num_threads(num_threads);

    neighbors.resize(n*k);
    dists.resize(n*k);
    found.resize(n);

    Index nz = 0;

    #pragma omp parallel for reduction(+:nz)
    for (Index i = 0; i < n; ++i)
    {
        found[i] = knn_query(points, &points[i*d], k, &neighbors[i*k], &dists[i*k]);
        nz += found[i];
    }

    return nz;
}

template <class Distance, class Real, class Atom>
void CoverTree<Distance, Real, Atom>::reorder_vertices(const Index *dfs_ordering)
{
    Index m = num_vertices();
    VertexVector new_vertices;

    IndexVector old_to_new(m);
    new_vertices.reserve(m);

    for (Index i = 0; i < m; ++i)
    {
        Vertex v = vertices[dfs_ordering[i]];
        old_to_new[dfs_ordering[i]] = i;
        new_vertices.push_back(v);
    }

    for (auto& v : new_vertices)
    {
        std::for_each(v.children.begin(), v.children.end(), [&](Index& id) { id = old_to_new[id]; });
    }

    std::swap(vertices, new_vertices);
}
