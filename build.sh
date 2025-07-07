#!/bin/bash
#

pushd bindings
mpic++ -fopenmp -O3 -Wall -shared -std=c++20 -undefined dynamic_lookup -fPIC -I/Users/gabrielraulet/miniconda3/include/python3.11 -I/Users/gabrielraulet/miniconda3/lib/python3.11/site-packages/pybind11/include -I/Users/gabrielraulet/miniconda3/lib/python3.11/site-packages/mpi4py/include metricspace.cpp -I../include -o metricspace.cpython-311-darwin.so
popd
