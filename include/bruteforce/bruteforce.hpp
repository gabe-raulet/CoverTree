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

#include "utils.h"

template <class Metric>
typename Metric::index_type
BruteForce<Metric>::radius_neighbors(Real radius, IndexVector& myneighs, RealVector& mydists, IndexVector& myptrs, MPI_Comm comm) const
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

    AtomVector curpoints(metric.point(0), metric.point(0) + metric.num_atoms());
    AtomVector nextpoints;

    TripleVector mytriples;
    IndexVector w(mysize, 0);

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

/*
 * Rank 0 starts with points[0..24]
 * Rank 1 starts with points[25..49]
 * Rank 2 starts with points[50..74]
 * Rank 3 starts with points[75..99]
 *
 * Step 0:
 *
 *     Rank 0 computes points[0..24] vs points[0..24] (0 vs 0)
 *     Rank 1 computes points[25..49] vs points[25..49] (1 vs 1)
 *     Rank 2 computes points[50..74] vs points[50..74] (2 vs 2)
 *     Rank 3 computes points[75..99] vs points[75..99] (3 vs 3)
 *
 * Step 1:
 *
 *     Rank 0 computes points[0..24] vs points[25..49] (0 vs 1)
 *     Rank 1 computes points[25..49] vs points[50..74] (1 vs 2)
 *     Rank 2 computes points[50..74] vs points[75..99] (2 vs 3)
 *     Rank 3 computes points[75..99] vs points[0..24] (3 vs 0)
 *
 * Step 2:
 *
 *     Rank 0 computes points[0..24] vs points[50..74] (0 vs 2)
 *     Rank 1 computes points[25..49] vs points[75..99] (1 vs 3)
 *     Rank 2 computes points[50..74] vs points[0..24] (2 vs 0)
 *     Rank 3 computes points[75..99] vs points[25..49] (3 vs 1)
 *
 * Step 3:
 *
 *     Rank 0 computes points[0..24] vs points[75..99] (0 vs 3)
 *     Rank 1 computes points[25..49] vs points[0..24] (1 vs 0)
 *     Rank 2 computes points[50..74] vs points[25..49] (2 vs 1)
 *     Rank 3 computes points[75..99] vs points[50..74] (3 vs 2)
 */
