import numpy as np


def min_edit_distance(source, target):
    n = len(source)
    m = len(target)
    matrix = np.zeros(shape=[n, m])
