import numpy as np
from bindings.metricspace import *

def MetricSpace(points, metric="euclidean"):
    if metric == "euclidean":
        if points.dtype == "float32": return EuclideanSpaceFloat(points)
        elif points.dtype == "float64": return EuclideanSpaceDouble(points)
        elif points.dtype == "uint8": return EuclideanSpaceUChar(points)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if points.dtype == "float32": return ManhattanSpaceFloat(points)
        elif points.dtype == "float64": return ManhattanSpaceDouble(points)
        elif points.dtype == "uint8": return ManhattanSpaceUChar(points)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if points.dtype == "float32": return ChebyshevSpaceFloat(points)
        elif points.dtype == "float64": return ChebyshevSpaceDouble(points)
        elif points.dtype == "uint8": return ChebyshevSpaceUChar(points)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

def BruteForce(space):
    kind = str(np.array(space, copy=False).dtype)
    metric = space.metric()
    if metric == "euclidean":
        if kind == "float32": return BruteForceEuclideanFloat(space)
        elif kind == "float64": return BruteForceEuclideanDouble(space)
        elif kind == "uint8": return BruteForceEuclideanUChar(space)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if kind == "float32": return BruteForceManhattanFloat(space)
        elif kind == "float64": return BruteForceManhattanDouble(space)
        elif kind == "uint8": return BruteForceManhattanUChar(space)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if kind == "float32": return BruteForceChebyshevFloat(space)
        elif kind == "float64": return BruteForceChebyshevDouble(space)
        elif kind == "uint8": return BruteForceChebyshevUChar(space)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

def CoverTree(space):
    kind = str(np.array(space, copy=False).dtype)
    metric = space.metric()
    if metric == "euclidean":
        if kind == "float32": return CoverTreeEuclideanFloat(space)
        elif kind == "float64": return CoverTreeEuclideanDouble(space)
        elif kind == "uint8": return CoverTreeEuclideanUChar(space)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if kind == "float32": return CoverTreeManhattanFloat(space)
        elif kind == "float64": return CoverTreeManhattanDouble(space)
        elif kind == "uint8": return CoverTreeManhattanUChar(space)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if kind == "float32": return CoverTreeChebyshevFloat(space)
        elif kind == "float64": return CoverTreeChebyshevDouble(space)
        elif kind == "uint8": return CoverTreeChebyshevUChar(space)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

