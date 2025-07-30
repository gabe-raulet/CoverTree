#!/bin/bash

export MPICH_MPIIO_DVS_MAXNODES=1

INFILE=$1
RADIUS=$2
CENTERS_PER_PROC=$3

for NUM_PROCS in 16 32 64 128
do
    srun -N 1 -n $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b static
    srun -N 1 -n $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b steal
    srun -N 1 -n $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b static
    srun -N 1 -n $NUM_PROCS ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b steal
    srun -N 1 -n $NUM_PROCS ./systolic $INFILE $RADIUS
    printf "\n"
done

for NUM_NODES in 2 4
do
    srun -N $NUM_NODES --ntasks-per-node 128 ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b static
    srun -N $NUM_NODES --ntasks-per-node 128 ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a cyclic -b steal
    srun -N $NUM_NODES --ntasks-per-node 128 ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b static
    srun -N $NUM_NODES --ntasks-per-node 128 ./rgraph -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -v0 -a multiway -b steal
    srun -N $NUM_NODES --ntasks-per-node 128 ./systolic $INFILE $RADIUS
    printf "\n"
done

