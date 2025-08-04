#1/bin/bash

if [[ $# -lt 3 ]] ; then
    echo 'Usage: ./download.sh <scratch> <points> <radius>'
    exit 1
fi

mkdir -p $1

mpirun -np 1 ./rgraph -i $2 -r $3 -M vor -A static -B static -v2 > $1/out.n1.static.static.txt
mpirun -np 2 ./rgraph -i $2 -r $3 -M vor -A static -B static -v2 > $1/out.n2.static.static.txt
mpirun -np 4 ./rgraph -i $2 -r $3 -M vor -A static -B static -v2 > $1/out.n4.static.static.txt
mpirun -np 8 ./rgraph -i $2 -r $3 -M vor -A static -B static -v2 > $1/out.n8.static.static.txt

mpirun -np 1 ./rgraph -i $2 -r $3 -M vor -A multiway -B static -v2 > $1/out.n1.multiway.static.txt
mpirun -np 2 ./rgraph -i $2 -r $3 -M vor -A multiway -B static -v2 > $1/out.n2.multiway.static.txt
mpirun -np 4 ./rgraph -i $2 -r $3 -M vor -A multiway -B static -v2 > $1/out.n4.multiway.static.txt
mpirun -np 8 ./rgraph -i $2 -r $3 -M vor -A multiway -B static -v2 > $1/out.n8.multiway.static.txt

mpirun -np 1 ./rgraph -i $2 -r $3 -M vor -A multiway -B steal -v2 > $1/out.n1.multiway.steal.txt
mpirun -np 2 ./rgraph -i $2 -r $3 -M vor -A multiway -B steal -v2 > $1/out.n2.multiway.steal.txt
mpirun -np 4 ./rgraph -i $2 -r $3 -M vor -A multiway -B steal -v2 > $1/out.n4.multiway.steal.txt
mpirun -np 8 ./rgraph -i $2 -r $3 -M vor -A multiway -B steal -v2 > $1/out.n8.multiway.steal.txt
