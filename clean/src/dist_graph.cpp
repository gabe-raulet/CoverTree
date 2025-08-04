#include "dist_graph.h"
#include <sstream>
#include <limits>
#include <assert.h>

DistGraph::DistGraph(MPI_Comm comm) : comm(comm), myptrs({0})
{
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
}

void DistGraph::add_neighbors(Index query, const IndexVector& neighbors, Index offset)
{
    myqueries.push_back(query);
    myneighs.reserve(myneighs.size() + neighbors.size());

    for (Index neighbor : neighbors)
        if (neighbor+offset != query)
            myneighs.push_back(neighbor+offset);

    myptrs.push_back(myneighs.size());
}

void DistGraph::write_edge_file(Index num_vertices, const char *filename) const
{
    std::ostringstream ss;
    Index my_num_edges = myneighs.size(), num_edges;

    MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);
    num_edges += num_vertices;

    if (!myrank)
    {
        ss << "% " << num_vertices << " " << num_vertices << " " << num_edges << "\n";
    }

    Index my_num_vertices = myqueries.size();

    for (Index i = 0; i < my_num_vertices; ++i)
        for (Index p = myptrs[i]; p < myptrs[i+1]; ++p)
            if (myqueries[i] != myneighs[p])
                ss << (myqueries[i]+1) << " " << (myneighs[p]+1) << "\n";

    Index mysize = num_vertices/nprocs;
    Index myleft = num_vertices%nprocs;

    if (myrank < myleft)
        mysize++;

    Index myoffset;
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    for (Index i = myoffset; i < myoffset+mysize; ++i)
        ss << (i+1) << " " << (i+1) << "\n";

    std::string s = ss.str();
    std::vector<char> buf(s.begin(), s.end());

    assert((buf.size() <= std::numeric_limits<int>::max()));

    MPI_Offset mycount = buf.size(), fileoffset, filesize;
    MPI_Exscan(&mycount, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    int truncate = 0;

    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &filesize);
    truncate = (filesize > 0);
    MPI_Bcast(&truncate, 1, MPI_INT, 0, comm);
    if (truncate) MPI_File_set_size(fh, 0);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
