#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A m4293
#SBATCH -q debug
#SBATCH --mail-user=gabe.h.raulet@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 30

export MPICH_MPIIO_DVS_MAXNODES=1

INFILE=$SCRATCH/points.normal.25K.d10.a200.fvecs

rm -rf $SCRATCH/normal_results
mkdir -p $SCRATCH/normal_results

RADIUS=2
CENTERS_PER_PROC=10

counter=1

for NUM_PROCS in 8 16 32 64 128
do
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M ct -v2 -j $SCRATCH/normal_results/ct.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M cvor -v2 -j $SCRATCH/normal_results/cvor.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B static -v2 -j $SCRATCH/normal_results/gvor.static.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B steal -v2 -j $SCRATCH/normal_results/gvor.steal.n$NUM_PROCS.json
done
