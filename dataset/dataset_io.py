import numpy as np
from urllib.request import urlopen, urlretrieve, build_opener, install_opener
import h5py
import os

opener = build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
install_opener(opener)

def renamefile(fname, ext):
    if fname.endswith(f".{ext}"): return fname
    return f"{fname.rstrip('.')}.{ext}"

def read_i8bin(fname, start=0, count=None):
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
        n = (n - start) if count is None else count
        arr = np.fromfile(f, count=n*d, dtype=np.int8, offset=start*d)
    return arr.reshape(n,d)

def read_u8bin(fname, start=0, count=None):
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
        n = (n - start) if count is None else count
        arr = np.fromfile(f, count=n*d, dtype=np.uint8, offset=start*d)
    return arr.reshape(n,d)

def read_fbin(fname, start=0, count=None):
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, count=2, dtype=np.uint32)
        n = (n - start) if count is None else count
        arr = np.fromfile(f, count=n*d, dtype=np.float32, offset=start*4*d)
    return arr.reshape(n,d)

def read_fvecs(fname, start=0, count=None):
    with open(fname, "rb") as f:
        n = os.path.getsize(fname)
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert n % (4*(d+1)) == 0
        n = (n//(4*(d+1)) - start) if count is None else count
        f.seek(0)
        arr = np.fromfile(f, count=n*(d+1), dtype=np.float32, offset=start*(4*(d+1)))
    return arr.reshape(-1,d+1)[:,1:].copy()

def read_ivecs(fname, start=0, count=None):
    return read_fvecs(fname, start, count).view(np.int32)

def read_bvecs(fname, start=0, count=None):
    with open(fname, "rb") as f:
        n = os.path.getsize(fname)
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert n % (d+1) == 0
        n = (n//(d+1) - start) if count is None else count
        f.seek(0)
        arr = np.fromfile(f, count=n*(d+1), dtype=np.uint8, offset=start*(d+1))
    return arr.reshape(-1,d+1)[:,1:].copy()

def write_i8bin(fname, points):
    fname = renamefile(fname, "i8bin")
    with open(fname, "wb") as f:
        n, d = points.shape
        np.array([n, d], dtype=np.uint32).tofile(f)
        points.tofile(f)

def write_u8bin(fname, points):
    fname = renamefile(fname, "u8bin")
    with open(fname, "wb") as f:
        n, d = points.shape
        np.array([n, d], dtype=np.uint32).tofile(f)
        points.tofile(f)

def write_fbin(fname, points):
    fname = renamefile(fname, "fbin")
    with open(fname, "wb") as f:
        n, d = points.shape
        np.array([n, d], dtype=np.uint32).tofile(f)
        points.tofile(f)


def write_fvecs(fname, points):
    fname = renamefile(fname, "fvecs")
    with open(fname, "wb") as f:
        n, d = points.shape
        arr = np.insert(points, 0, np.int32(d).view(np.float32), axis=1)
        arr.tofile(f)

def write_ivecs(fname, points):
    fname = renamefile(fname, "ivecs")
    with open(fname, "wb") as f:
        n, d = points.shape
        arr = np.insert(points, 0, np.int32(d), axis=1)
        arr.tofile(f)

def read_file(fname, start=0, count=None):
    if fname.endswith("i8bin"): return read_i8bin(fname, start, count)
    elif fname.endswith("u8bin"): return read_u8bin(fname, start, count)
    elif fname.endswith("fbin"): return read_fbin(fname, start, count)
    elif fname.endswith("fvecs"): return read_fvecs(fname, start, count)
    elif fname.endswith("ivecs"): return read_ivecs(fname, start, count)
    elif fname.endswith("bvecs"): return read_bvecs(fname, start, count)
    else: return None
    #  else: raise Exception(f"cannot read file '{fname}': unknown extension")

def write_file(fname, points):
    if fname.endswith("i8bin"): write_i8bin(fname, points)
    elif fname.endswith("u8bin"): write_u8bin(fname, points)
    elif fname.endswith("fbin"): write_fbin(fname, points)
    elif fname.endswith("fvecs"): write_fvecs(fname, points)
    elif fname.endswith("ivecs"): write_ivecs(fname, points)
    else: raise Exception(f"cannot read file '{fname}': unknown extension")

#  def download(url, dest):
    #  if not os.path.exists(dest):
        #  print(f"downloading {url} -> {dest}...")
        #  urlretrieve(url, dest)

#  download("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin", "datasets/query.public.10K.u8bin")
#  download("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_public_queries.u8bin", "datasets/FB_ssnpp_public_queries.u8bin")

#  points = read_file("datasets/FB_ssnpp_public_queries.u8bin")
#  write_file("example.u8bin", points)

#  points = read_u8bin("query.public.10K.u8bin")
#  write_u8bin("test.", points)
#  points = read_fvecs("corel.fvecs")
