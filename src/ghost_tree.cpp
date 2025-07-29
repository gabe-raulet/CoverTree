#include "ghost_tree.h"

void GhostTreeHeader::create_header_type(MPI_Datatype *MPI_GHOST_TREE_HEADER)
{
    GhostTreeHeader dummy;
    int blklens[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_INDEX, MPI_INDEX, MPI_INDEX, MPI_INDEX, MPI_INDEX};
    MPI_Aint disps[5];

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&(dummy.id), &disps[0]);
    MPI_Get_address(&(dummy.cur_query), &disps[1]);
    MPI_Get_address(&(dummy.num_queries), &disps[2]);
    MPI_Get_address(&(dummy.num_points), &disps[3]);
    MPI_Get_address(&(dummy.num_vertices), &disps[4]);

    for (int i = 0; i < 5; ++i) disps[i] -= base_address;
    MPI_Type_create_struct(5, blklens, disps, types, MPI_GHOST_TREE_HEADER);
    MPI_Type_commit(MPI_GHOST_TREE_HEADER);
}

Index GhostTree::make_queries(Index count, Real radius, IndexVector& neighs, IndexVector& queries, IndexVector& ptrs, Index& queries_made)
{
    Index edges_found = 0;

    if (finished()) return edges_found;

    if (count < 0) count = header.num_queries - header.cur_query;
    else count = std::min(count, header.num_queries - header.cur_query);

    queries_made = count;

    for (Index i = 0; i < count; ++i, ++header.cur_query)
    {
        Index found = tree.radius_query_indexed(points, header.cur_query, radius, neighs, indices);
        queries.push_back(indices[header.cur_query]);
        ptrs.push_back(neighs.size());
        edges_found += found;
    }

    return edges_found;
}

GhostTree::GhostTree(const CoverTree& tree, const PointVector& points, const IndexVector& indices, Index num_queries, Index id)
    : tree(tree),
      points(points),
      indices(indices),
      header(id, 0, num_queries, points.num_points(), tree.num_vertices()) {}


void GhostTree::allocate(const GhostTreeHeader& recv_header, int dim)
{
    header = recv_header;

    tree.allocate(header.num_vertices);
    points.resize(header.num_points, dim);
    indices.resize(header.num_points);
}

void GhostTree::isend(int dest, MPI_Comm comm, std::vector<MPI_Request>& reqs)
{
    int m = header.num_vertices;
    int n = header.num_points;
    int tag = header.id;

    const void* bufs[6];
    int counts[6];
    MPI_Datatype dtypes[6];

    bufs[0] = tree.childarr.data();  counts[0] = m-1;                dtypes[0] = MPI_INDEX;
    bufs[1] = tree.childptrs.data(); counts[1] = m+1;                dtypes[1] = MPI_INDEX;
    bufs[2] = tree.centers.data();   counts[2] = m;                  dtypes[2] = MPI_INDEX;
    bufs[3] = indices.data();        counts[3] = n;                  dtypes[3] = MPI_INDEX;
    bufs[4] = tree.radii.data();     counts[4] = m;                  dtypes[4] = MPI_REAL;
    bufs[5] = points.data();         counts[5] = points.num_atoms(); dtypes[5] = MPI_ATOM;

    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[0], counts[0], dtypes[0], dest, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[1], counts[1], dtypes[1], dest, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[2], counts[2], dtypes[2], dest, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[3], counts[3], dtypes[3], dest, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[4], counts[4], dtypes[4], dest, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Isend(bufs[5], counts[5], dtypes[5], dest, tag, comm, &reqs.back());
}

void GhostTree::irecv(int source, MPI_Comm comm, std::vector<MPI_Request>& reqs)
{
    int m = header.num_vertices;
    int n = header.num_points;
    int tag = header.id;

    void* bufs[6];
    int counts[6];
    MPI_Datatype dtypes[6];

    bufs[0] = tree.childarr.data();  counts[0] = m-1;                dtypes[0] = MPI_INDEX;
    bufs[1] = tree.childptrs.data(); counts[1] = m+1;                dtypes[1] = MPI_INDEX;
    bufs[2] = tree.centers.data();   counts[2] = m;                  dtypes[2] = MPI_INDEX;
    bufs[3] = indices.data();        counts[3] = n;                  dtypes[3] = MPI_INDEX;
    bufs[4] = tree.radii.data();     counts[4] = m;                  dtypes[4] = MPI_REAL;
    bufs[5] = points.data();         counts[5] = points.num_atoms(); dtypes[5] = MPI_ATOM;

    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[0], counts[0], dtypes[0], source, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[1], counts[1], dtypes[1], source, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[2], counts[2], dtypes[2], source, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[3], counts[3], dtypes[3], source, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[4], counts[4], dtypes[4], source, tag, comm, &reqs.back());
    reqs.push_back(MPI_REQUEST_NULL); MPI_Irecv(bufs[5], counts[5], dtypes[5], source, tag, comm, &reqs.back());
}
