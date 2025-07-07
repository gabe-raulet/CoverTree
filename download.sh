#!/bin/bash

shopt -s nullglob

download_file() {
    local url="$1"
    wget -q --show-progress "$url" || echo "Failed to download: $url"
}

download_files() {
    for opt in "${@:1}" ; do
        download_file $opt
    done
}

ARTIFICIAL40="https://portal.nersc.gov/project/m1982/rgraph_datasets/artificial40.fvecs"
COREL="https://portal.nersc.gov/project/m1982/rgraph_datasets/corel.fvecs"
FACES="https://portal.nersc.gov/project/m1982/rgraph_datasets/faces.fvecs"
COVTYPE="https://portal.nersc.gov/project/m1982/rgraph_datasets/covtype.fvecs"
TWITTER="https://portal.nersc.gov/project/m1982/rgraph_datasets/twitter.fvecs"
SIFTSMALL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFT="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
BIGANN="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
FACEBOOK="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_public_queries.u8bin"
DEEP="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin"
T2I="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin"
FASHION_MNIST="https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"

if [[ $# -lt 1 ]] ; then
    echo 'No scratch directory path provided!'
    echo 'Usage: ./download.sh <scratch> [datasets...]'
    exit 1
fi

mkdir -p $1/datasets
pushd $1/datasets

for opt in "${@:2}" ; do
    if [[ "$opt" == "sift" ]] ; then
        download_file $SIFT
        tar xvzf sift.tar.gz && rm sift.tar.gz
        mv sift/*.fvecs . && rm -rf sift
    fi
    if [[ "$opt" == "siftsmall" ]] ; then
        download_file $SIFTSMALL
        tar xvzf siftsmall.tar.gz && rm siftsmall.tar.gz
        mv siftsmall/*.fvecs . && rm -rf siftsmall
    fi
    if [[ "$opt" == "bigann" ]] ; then
        download_file $BIGANN
        mv query.public.10K.u8bin bigann.u8bin
    fi
    if [[ "$opt" == "facebook" ]] ; then
        download_file $FACEBOOK
        mv FB_ssnpp_public_queries.u8bin facebook.u8bin
    fi
    if [[ "$opt" == "deep" ]] ; then
        download_file $DEEP
        mv query.public.10K.fbin deep.fbin
    fi
    if [[ "$opt" == "t2i" ]] ; then
        download_file $T2I
        mv query.public.100K.fbin t2i.fbin
    fi
    if [[ "$opt" == "fashion_mnist" ]] ; then
        download_file $FASHION_MNIST
    fi
    if [[ "$opt" == "artificial40" ]] ; then
        download_file $ARTIFICIAL40
    fi
    if [[ "$opt" == "corel" ]] ; then
        download_file $COREL
    fi
    if [[ "$opt" == "faces" ]] ; then
        download_file $FACES
    fi
    if [[ "$opt" == "covtype" ]] ; then
        download_file $COVTYPE
    fi
    if [[ "$opt" == "twitter" ]] ; then
        download_file $TWITTER
    fi
    if [[ "$opt" == "mlpack_small" ]] ; then
        download_files $ARTIFICIAL40 $COREL $FACES
    fi
    if [[ "$opt" == "mlpack_large" ]] ; then
        download_files $COVTYPE $TWITTER
    fi
    if [[ "$opt" == "mlpack" ]] ; then
        download_files $ARTIFICIAL40 $COREL $FACES $COVTYPE $TWITTER
    fi
done

popd

python dataset_io.py $1/datasets/* | column -t
