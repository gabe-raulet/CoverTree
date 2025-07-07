template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_query(const Atom *query, Real radius, IndexVector& neighs, RealVector& dists) const
{
    Index found = 0;
    Index n = metric.num_points();

    for (Index i = 0; i < n; ++i)
    {
        Real dist = metric.distance(i, query);

        if (dist <= radius)
        {
            neighs.push_back(i);
            dists.push_back(dist);
            found++;
        }
    }

    return found;
}

template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_neighbors(const Atom *queries, Index num_queries, Real radius, IndexVector& neighs, RealVector& dists, IndexVector& ptrs) const
{
    neighs.clear();
    dists.clear();
    ptrs.resize(num_queries+1);

    Index i;
    Index d = num_dimensions();
    const Atom *query = queries;

    for (i = 0; i < num_queries; ++i, query += d)
    {
        ptrs[i] = neighs.size();
        radius_query(query, radius, neighs, dists);
    }

    ptrs[num_queries] = neighs.size();
    return ptrs[num_queries];
}
