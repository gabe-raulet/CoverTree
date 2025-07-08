template <class Metric> typename Metric::index_type
NearestNeighbors<Metric>::radius_neighbors(const Atom **queries, Index num_queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const
{
    neighs.clear();
    dists.clear();
    ptrs.resize(num_queries+1);

    if (num_threads == 1)
    {
        for (Index i = 0; i < num_queries; ++i)
        {
            ptrs[i] = neighs.size();
            radius_query(queries[i], radius, dists, neighs);
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
            Index count = radius_query(queries[i], radius, dists_vector[i], neighs_vector[i]);
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

template <class Metric> typename Metric::index_type
NearestNeighbors<Metric>::radius_neighbors(const Atom *queries, Index num_queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const
{
    Index d = num_dimensions();
    std::vector<const Atom*> query_ptrs(num_queries);

    for (Index i = 0; i < num_queries; ++i, queries += d)
        query_ptrs[i] = queries;

    return radius_neighbors(query_ptrs.data(), num_queries, radius, dists, neighs, ptrs, num_threads);
}

template <class Metric> typename Metric::index_type
NearestNeighbors<Metric>::radius_neighbors(const IndexVector& queries, Real radius, RealVector& dists, IndexVector& neighs, IndexVector& ptrs, int num_threads) const
{
    Index num_queries = queries.size();
    std::vector<const Atom*> query_ptrs(num_queries);

    for (Index i = 0; i < num_queries; ++i)
        query_ptrs[i] = metric[i];

    return radius_neighbors(query_ptrs.data(), num_queries, radius, dists, neighs, ptrs, num_threads);
}

template <class Metric> typename Metric::index_type
NearestNeighbors<Metric>::radius_neighbors(Real radius, RealVector& mydists, IndexVector& myneighs, IndexVector& myptrs, MPI_Comm comm) const
{
    Index dim = num_dimensions();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_INDEX = mpi_type<Index>();
    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    IndexVector allsizes(nprocs), alloffsets(nprocs);
    allsizes[myrank] = num_points();

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, allsizes.data(), 1, MPI_INDEX, comm);

    std::exclusive_scan(allsizes.begin(), allsizes.end(), alloffsets.begin(), (Index)0);

    Index mysize = allsizes[myrank];

    MPI_Request reqs[2];

    int next = (myrank+1)%nprocs;
    int prev = (myrank-1+nprocs)%nprocs;
    int cur = myrank;

    AtomVector curpoints(metric.data(), metric.data() + metric.num_atoms());
    AtomVector nextpoints;

    TripleVector mytriples;
    IndexVector w(mysize, 0);

    IndexVector neighs, ptrs;
    RealVector dists;

    for (int step = 0; step < nprocs; ++step)
    {
        Index numrecv = allsizes[(cur+1)%nprocs];
        Index numsend = allsizes[cur];

        int recvcount = numrecv*dim;
        int sendcount = numsend*dim;

        nextpoints.resize(recvcount);

        MPI_Irecv(nextpoints.data(), recvcount, MPI_ATOM, next, myrank, comm, &reqs[0]);
        MPI_Isend(curpoints.data(), sendcount, MPI_ATOM, prev, prev, comm, &reqs[1]);

        Index curoffset = alloffsets[cur];
        Index cursize = allsizes[cur];

        for (Index i = 0; i < mysize; ++i)
            for (Index j = 0; j < cursize; ++j)
            {
                Real dij = metric.distance(i, &curpoints[j*dim]);

                if (dij <= radius)
                {
                    mytriples.emplace_back(i, j+curoffset, dij);
                    w[i]++;
                }
            }

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        cur = (cur+1)%nprocs;
        std::swap(curpoints, nextpoints);
    }

    Index nz = 0;
    myptrs.resize(mysize+1);

    for (Index i = 0; i < mysize; ++i)
    {
        myptrs[i] = nz;
        nz += w[i];
        w[i] = myptrs[i];
    }

    myptrs[mysize] = nz;
    mydists.resize(nz);
    myneighs.resize(nz);

    for (const auto& [i, j, dij] : mytriples)
    {
        Index p = w[i]++;
        myneighs[p] = j;
        mydists[p] = dij;
    }

    MPI_Allreduce(MPI_IN_PLACE, &nz, 1, MPI_INDEX, MPI_SUM, comm);

    return nz;
}
