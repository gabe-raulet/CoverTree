import os
import time
import tarfile
from dataset_io import *
import json

def download(url, dest):
    if not os.path.exists(dest):
        print(f"downloading {url} -> {dest}...")
        urlretrieve(url, dest)
    else:
        print("Already exists")

def untar(tarname):
    assert tarname.endswith("tar.gz")
    dname = tarname.rstrip(".tar.gz")
    if not os.path.exists(dname):
        tar = tarfile.open(tarname, "r:gz")
        tar.extractall()
        tar.close()
    else:
        print("Already exists")


download("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin", "bigann.query.10K.u8bin")
download("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_public_queries.u8bin", "facebook.query.u8bin")
download("https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin", "deep.query.10K.fbin")
download("https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin", "t2i.query.100K.fbin")
download("https://portal.nersc.gov/project/m1982/rgraph_datasets/artificial40.fvecs", "artificial40.fvecs")
download("https://portal.nersc.gov/project/m1982/rgraph_datasets/corel.fvecs", "corel.fvecs")
download("https://portal.nersc.gov/project/m1982/rgraph_datasets/faces.fvecs", "faces.fvecs")
download("https://portal.nersc.gov/project/m1982/rgraph_datasets/covtype.fvecs", "covtype.fvecs")
download("https://portal.nersc.gov/project/m1982/rgraph_datasets/twitter.fvecs", "twitter.fvecs")

#  download("https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "datasets/fashion-mnist-784-euclidean.hdf5")

#  download("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz", "siftsmall.tar.gz")
#  download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", "sift.tar.gz")

#  untar("siftsmall.tar.gz")
#  untar("sift.tar.gz")

