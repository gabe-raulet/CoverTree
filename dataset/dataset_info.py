from pathlib import Path
from dataset_io import *
import sys

def show_info(fname):
    points = read_file(fname)
    if points is not None:
        n, d = points.shape
        print(f"{n}\t{d}\t{points.dtype}\t{fname}")

print("num_points\tdimensions\tdtype\tfilename")
for p in Path(".").iterdir():
    if p.is_file():
        show_info(str(p))
