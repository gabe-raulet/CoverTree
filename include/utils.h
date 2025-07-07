#ifndef UTILS_H_
#define UTILS_H_

#include <typeinfo>
#include <stdexcept>
#include <mpi.h>

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

#endif
