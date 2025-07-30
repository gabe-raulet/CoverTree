#!/bin/bash

INFILE=$1
RADIUS=$2
CENTERS_PER_PROC=$3

for NUM_PROCS in 1 2 4 8 12
do
    mpirun -np $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b static
    mpirun -np $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b steal
    mpirun -np $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b static
    mpirun -np $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b steal
    mpirun -np $NUM_PROCS ./systolic $INFILE $RADIUS
done

