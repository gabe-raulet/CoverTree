#!/bin/bash

#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -A m4293
#SBATCH -q debug
#SBATCH --mail-user=gabe.h.raulet@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 30

export MPICH_MPIIO_DVS_MAXNODES=1

INFILE=$SCRATCH/scratch/datasets/covtype.fvecs

mkdir -p $SCRATCH/covtype_results

RADIUS=150
CENTERS_PER_PROC=10

counter=1

for NUM_PROCS in 8 16 32 64 128
do
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M ct -v2 -j $SCRATCH/covtype_results/ct.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M cvor -v2 -j $SCRATCH/covtype_results/cvor.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B static -v2 -j $SCRATCH/covtype_results/gvor.static.n$NUM_PROCS.json
    srun -N 1 -n $NUM_PROCS ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B steal -v2 -j $SCRATCH/covtype_results/gvor.steal.n$NUM_PROCS.json
    printf "\n"
done

for NUM_NODES in 2 4 8
do
    srun -N $NUM_NODES --ntasks-per-node=128 ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M ct -v2 -j $SCRATCH/covtype_results/ct.n$((128*NUM_NODES)).json
    srun -N $NUM_NODES --ntasks-per-node=128 ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M cvor -v2 -j $SCRATCH/covtype_results/cvor.n$((128*NUM_NODES)).json
    srun -N $NUM_NODES --ntasks-per-node=128 ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B static -v2 -j $SCRATCH/covtype_results/gvor.static.n$((128*NUM_NODES)).json
    srun -N $NUM_NODES --ntasks-per-node=128 ./rgraph.py -i $INFILE -r $RADIUS -m $CENTERS_PER_PROC -M gvor -B steal -v2 -j $SCRATCH/covtype_results/gvor.steal.n$((128*NUM_NODES)).json
    printf "\n"
done
