import numpy as np
from bindings.metricspace import *

def MetricSpace(points, metric="euclidean"):
    if metric == "euclidean":
        if points.dtype == "float32": return EuclideanSpaceFloat(points)
        elif points.dtype == "float64": return EuclideanSpaceDouble(points)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if points.dtype == "float32": return ManhattanSpaceFloat(points)
        elif points.dtype == "float64": return ManhattanSpaceDouble(points)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if points.dtype == "float32": return ChebyshevSpaceFloat(points)
        elif points.dtype == "float64": return ChebyshevSpaceDouble(points)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

def BruteForce(space):
    kind = str(np.array(space, copy=False).dtype)
    metric = space.metric()
    if metric == "euclidean":
        if kind == "float32": return BruteForceEuclideanFloat(space)
        elif kind == "float64": return BruteForceEuclideanDouble(space)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if kind == "float32": return BruteForceManhattanFloat(space)
        elif kind == "float64": return BruteForceManhattanDouble(space)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if kind == "float32": return BruteForceChebyshevFloat(space)
        elif kind == "float64": return BruteForceChebyshevDouble(space)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

def CoverTree(space):
    kind = str(np.array(space, copy=False).dtype)
    metric = space.metric()
    if metric == "euclidean":
        if kind == "float32": return CoverTreeEuclideanFloat(space)
        elif kind == "float64": return CoverTreeEuclideanDouble(space)
        else: raise Exception("Not implemented")
    elif metric == "manhattan":
        if kind == "float32": return CoverTreeManhattanFloat(space)
        elif kind == "float64": return CoverTreeManhattanDouble(space)
        else: raise Exception("Not implemented")
    elif metric == "chebyshev":
        if kind == "float32": return CoverTreeChebyshevFloat(space)
        elif kind == "float64": return CoverTreeChebyshevDouble(space)
        else: raise Exception("Not implemented")
    else:
        raise Exception("Not implemented")

