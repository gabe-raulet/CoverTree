#ifndef UTILS_H_
#define UTILS_H_

#include <concepts>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <tuple>
#include <mpi.h>

using Index = int64_t;
using Real = float;
using Atom = float;

using Tuple = std::tuple<Index, Real>;
using Triple = std::tuple<Index, Index, Real>;
using IndexPair = std::tuple<Index, Index>;

using IndexVector = std::vector<Index>;
using RealVector = std::vector<Real>;
using AtomVector = std::vector<Atom>;
using IndexQueue = std::deque<Index>;
using IndexMap = std::unordered_map<Index, Index>;
using TupleVector = std::vector<Tuple>;
using TripleVector = std::vector<Triple>;
using IndexPairVector = std::vector<IndexPair>;

using PointIter = typename AtomVector::const_iterator;
using IndexIter = typename IndexVector::const_iterator;

template <class T>
MPI_Datatype mpi_type()
{
    if      constexpr (std::same_as<T, char>)               return MPI_CHAR;
    else if constexpr (std::same_as<T, signed char>)        return MPI_SIGNED_CHAR;
    else if constexpr (std::same_as<T, short>)              return MPI_SHORT;
    else if constexpr (std::same_as<T, int>)                return MPI_INT;
    else if constexpr (std::same_as<T, long>)               return MPI_LONG;
    else if constexpr (std::same_as<T, long long>)          return MPI_LONG_LONG;
    else if constexpr (std::same_as<T, unsigned char>)      return MPI_UNSIGNED_CHAR;
    else if constexpr (std::same_as<T, unsigned short>)     return MPI_UNSIGNED_SHORT;
    else if constexpr (std::same_as<T, unsigned int>)       return MPI_UNSIGNED;
    else if constexpr (std::same_as<T, unsigned long>)      return MPI_UNSIGNED_LONG;
    else if constexpr (std::same_as<T, unsigned long long>) return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::same_as<T, float>)              return MPI_FLOAT;
    else if constexpr (std::same_as<T, double>)             return MPI_DOUBLE;
    else if constexpr (std::same_as<T, long double>)        return MPI_LONG_DOUBLE;
    else if constexpr (std::same_as<T, bool>)               return MPI_CXX_BOOL;
    else
        throw std::runtime_error("not implemented");
}

template <class Iter>
std::string container_repr(Iter first, Iter last)
{
    std::stringstream ss;
    ss << "[";

    while (first != last)
    {
        ss << *first;
        first++;

        if (first != last) ss << ", ";
    }

    ss << "]";
    return ss.str();
}

#ifdef MPI_INDEX
#undef MPI_INDEX
#endif

#define MPI_INDEX MPI_INT64_T

#ifdef MPI_REAL
#undef MPI_REAL
#endif

#define MPI_REAL MPI_FLOAT

#ifdef MPI_ATOM
#undef MPI_ATOM
#endif

#define MPI_ATOM MPI_FLOAT

#endif
