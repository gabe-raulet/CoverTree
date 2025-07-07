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
BruteForce<Metric>::radius_neighbors(const Atom **queries, Index num_queries, Real radius, IndexVector& neighs, RealVector& dists, IndexVector& ptrs, int num_threads) const
{
    neighs.clear();
    dists.clear();
    ptrs.resize(num_queries+1);

    if (num_threads == 1)
    {
        for (Index i = 0; i < num_queries; ++i)
        {
            ptrs[i] = neighs.size();
            radius_query(queries[i], radius, neighs, dists);
        }

        ptrs[num_queries] = neighs.size();
    }
    else
    {
        std::vector<IndexVector> neighs_vector(num_queries);
        std::vector<RealVector> dists_vector(num_queries);

        Index nz = 0;

        #pragma omp parallel for reduction(+:nz) num_threads(num_threads)
        for (Index i = 0; i < num_queries; ++i)
        {
            Index count = radius_query(queries[i], radius, neighs_vector[i], dists_vector[i]);
            nz += count;
        }

        neighs.reserve(nz);
        dists.reserve(nz);

        for (Index i = 0; i < num_queries; ++i)
        {
            ptrs[i] = neighs.size();

            std::copy(neighs_vector[i].begin(), neighs_vector[i].end(), std::back_inserter(neighs));
            std::copy(dists_vector[i].begin(), dists_vector[i].end(), std::back_inserter(dists));
        }

        ptrs[num_queries] = neighs.size();
    }

    return ptrs[num_queries];
}

template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_neighbors(const Atom *queries, Index num_queries, Real radius, IndexVector& neighs, RealVector& dists, IndexVector& ptrs, int num_threads) const
{
    Index d = num_dimensions();
    std::vector<const Atom*> query_ptrs(num_queries);

    for (Index i = 0; i < num_queries; ++i, queries += d)
        query_ptrs[i] = queries;

    return radius_neighbors(query_ptrs.data(), num_queries, radius, neighs, dists, ptrs, num_threads);
}

template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_neighbors(const IndexVector& queries, Real radius, IndexVector& neighs, RealVector& dists, IndexVector& ptrs, int num_threads) const
{
    Index num_queries = queries.size();
    std::vector<const Atom*> query_ptrs(num_queries);

    for (Index i = 0; i < num_queries; ++i)
        query_ptrs[i] = metric[i];

    return radius_neighbors(query_ptrs.data(), num_queries, radius, neighs, dists, ptrs, num_threads);
}
