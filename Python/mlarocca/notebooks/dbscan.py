import heapq
import math
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers

from collections import Counter
from itertools import chain, groupby, zip_longest
from matplotlib import pyplot as plt
from operator import itemgetter
from random import randrange, random, sample
from scipy.spatial import KDTree
from statistics import mean


from typing import Callable, List, Optional, Tuple

NOISE = -1

def dbscan(points: List[Tuple], eps: float, min_points: int) -> List[int]:
    """DBSCAN clustering.

    Args:
        points: A list of points to cluster.
        eps: The radius `epsilon` of the dense regions.
        min_points: The minimum number of points that needs to be within a distance `eps` for a point to be a core point.

    Returns:
        A list of the cluster indices for each point.
    """
    n = len(points)
    cluster_indices = [None] * n
    current_index = 0
    kd_tree = KDTree(points)
    for i in range(n):
        if cluster_indices[i] is not None:
            continue
        process_set = {i}
        cluster_indices[i] = NOISE
        current_index += 1
        while len(process_set) > 0:
            j = process_set.pop()
            _, neighbors = kd_tree.query(points[j], k=None, distance_upper_bound=eps)
            if len(neighbors) < min_points:
                continue
            cluster_indices[j] = current_index
            process_set |= set(filter(lambda p: cluster_indices[p] is None or (p != j and cluster_indices[p] == NOISE),
                                  neighbors))
    return cluster_indices


def create_spherical_cluster(centroid, radius, n_points):
    def random_point_in_circle():
        alpha = random() * 2 * math.pi
        r = radius * math.sqrt(random())
        # r is dim-dimensional root of radius

        x = centroid[0] + r * math.cos(alpha)
        y = centroid[1] + r * math.sin(alpha)
        return (x, y)

    return [random_point_in_circle() for _ in range(n_points)]