#include "dist_query.h"


DistQuery::DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<CellVector>& my_cell_vectors, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, MPI_Comm comm)
    : radius(radius),
      comm(comm),
      num_local_trees_completed(0),
      num_local_queries_found(0),
      num_local_edges_found(0)
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index s = mytrees.size();

    for (Index i = 0; i < s; ++i)
    {
        myqueue.emplace_back(mytrees[i], my_cell_vectors[i], my_query_sizes[i], mycells[i]);
    }

    MPI_Allreduce(&s, &num_global_trees, 1, MPI_INDEX, MPI_SUM, comm);
}

void DistQuery::static_balancing()
{
    
}
