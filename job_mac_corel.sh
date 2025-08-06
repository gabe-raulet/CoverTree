#!/bin/bash

RADIUS=0.15
CENTERS_PER_PROC=10
INFILE=scratch/datasets/corel.fvecs
RESULTS=mac_corel_results

rm -rf $RESULTS
mkdir -p $RESULTS

for NUM_PROCS in 1 2 4 8
do
    mpirun -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M ct -v2 -j $RESULTS/ct.n$NUM_PROCS.json; printf "\n"
    mpirun -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M cvor -v2 -j $RESULTS/cvor.n$NUM_PROCS.json; printf "\n"
    mpirun -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B static -v2 -j $RESULTS/gvor.static.n$NUM_PROCS.json; printf "\n"
    mpirun -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B steal -v2 -j $RESULTS/gvor.steal.n$NUM_PROCS.json; printf "\n"
done
