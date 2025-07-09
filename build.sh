#!/bin/bash

export INCLUDE_DIR=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/include/python3.11
export MPI4PY_DIR=/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/lib/python3.11/site-packages/mpi4py/include

pushd bindings
CC -fopenmp -O3 -Wall -shared -std=c++20 -fPIC -I$INCLUDE_DIR -I$INCLUDE_DIR/include -I$MPI4PY_DIR metricspace.cpp -I../include -o metricspace.so
#mpic++ -fopenmp -O3 -Wall -shared -std=c++20 -undefined dynamic_lookup -fPIC -I/Users/gabrielraulet/miniconda3/include/python3.11 -I/Users/gabrielraulet/miniconda3/lib/python3.11/site-packages/pybind11/include -I/Users/gabrielraulet/miniconda3/lib/python3.11/site-packages/mpi4py/include metricspace.cpp -I../include -o metricspace.cpython-311-darwin.so
popd
