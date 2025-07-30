#ifndef DIST_QUERY_H_
#define DIST_QUERY_H_

#include <mpi.h>
#include <deque>
#include "ghost_tree.h"

class DistQuery
{
    public:

        DistQuery(const std::vector<CoverTree>& mytrees, const std::vector<PointVector>& my_cell_vectors, const std::vector<IndexVector>& my_cell_indices, const IndexVector& my_query_sizes, const IndexVector& mycells, Real radius, int dim, MPI_Comm comm, int verbosity);

        void static_balancing();
        void random_shuffling(Index queries_per_tree);
        void random_stealing(Index queries_per_tree);

        Index my_edges_found() const { return num_local_edges_found; }

        void write_to_file(const char *fname) const;

    private:

        Real radius;
        std::deque<GhostTree> myqueue;

        IndexVector myneighs;
        IndexVector myqueries;
        IndexVector myptrs;

        Index num_global_trees;
        Index num_local_trees_completed;
        Index num_local_queries_made;
        Index num_local_edges_found;

        MPI_Comm comm;
        int myrank, nprocs;
        int dim;
        int verbosity;

        bool make_tree_queries(GhostTree& tree, Index count);

        void report_finished(double mytime);
        void report_finished(double mycomptime, double mycommtime);

        void shuffle_queues();
};

#endif
