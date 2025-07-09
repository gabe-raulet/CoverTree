#!/bin/bash

#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:60:00

mkdir -p stats

# TWITTER

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 4.5 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.twitter.euclidean.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 4.5 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.twitter.euclidean.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 20 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.twitter.manhattan.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 20 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.twitter.manhattan.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 1.15 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.twitter.chebyshev.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/twitter.fvecs -r 1.15 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.twitter.chebyshev.$((NODES*128)).json
done

# COVTYPE

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 200 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.covtype.euclidean.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 200 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.covtype.euclidean.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 300 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.covtype.manhattan.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 300 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.covtype.manhattan.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 75 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.covtype.chebyshev.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/covtype.fvecs -r 75 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.covtype.chebyshev.$((NODES*128)).json
done

# T2I

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 0.7 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.t2i.euclidean.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 0.7 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.t2i.euclidean.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 10 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.t2i.manhattan.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 10 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.t2i.manhattan.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 0.175 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.t2i.chebyshev.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/t2i.fbin -r 0.175 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.t2i.chebyshev.$((NODES*128)).json
done

# DEEP

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 0.95 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.deep.euclidean.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 0.95 -A covertree -m euclidean -l 15 -c 1.8 -j stats/stats.deep.euclidean.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 1 --ntasks-per-node=$PROCS --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 7.5 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.deep.manhattan.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N $NODES --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 7.5 -A covertree -m manhattan -l 15 -c 1.8 -j stats/stats.deep.manhattan.$((NODES*128)).json
done

for PROCS in 4 8 16 32 64
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 0.25 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.deep.chebyshev.$PROCS.json
done

for NODES in 1 2 4 8
do
    srun -N 4 --ntasks-per-node=128 --cpu_bind=cores ./rgraph_mpi.py -i scratch/datasets/deep.fbin -r 0.25 -A covertree -m chebyshev -l 15 -c 1.8 -j stats/stats.deep.chebyshev.$((NODES*128)).json
done
