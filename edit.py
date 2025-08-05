#!/usr/bin/env python3

from dataset_io import *

import numpy as np
import math
import sys
import getopt

start=0
count=None

def usage():
    global start, count
    sys.stderr.write(f"Usage: {sys.argv[0]} [options] -i <infile> -o <outfile>\n")
    sys.stderr.write(f"Options: -n INT   number of points [{'all' if not count else str(count)}]\n")
    sys.stderr.write(f"         -s INT   start offset [{start}]\n")
    sys.stderr.write(f"         -h       help message\n")
    sys.stderr.flush()
    sys.exit(1)

if __name__ == "__main__":

    infile = None
    outfile = None

    try: opts, args = getopt.getopt(sys.argv[1:], "i:n:s:o:h")
    except getopt.GetoptError as err: usage()

    for o, a in opts:
        if o == "-i": infile = a
        elif o == "-n": count = int(a)
        elif o == "-s": start = int(a)
        elif o == "-o": outfile = a
        elif o == "-h": usage()
        else: assert False, "unhandled option"

    if infile is None or outfile is None: usage()

    points = read_file(infile, start, count)
    write_file(outfile, points)

    sys.exit(0)
