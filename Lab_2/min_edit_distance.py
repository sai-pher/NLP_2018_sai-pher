import sys

import numpy as np


def del_cost(source):
    """
    Cost to delete a letter from a given string.

    :param source: The proposed letter to delete.
    :return: The cost to delete
    """

    # future proofing: created to allow for unique implementation edits at latter point.
    return 1


def ins_cost(source):
    """
    Cost to insert a letter from a given string.

    :param source: The proposed letter to insert.
    :return: The cost to delete
    """

    # future proofing: created to allow for unique implementation edits at latter point.
    return 1


def sub_cost(source, target):
    """
    Cost to substitute a letter from a given string.

    :param source: The proposed letter to substitute.
    :param target: The proposed letter to substituted.
    :return: The cost to substitute
    """

    # future proofing: created to allow for unique implementation edits at latter point.
    if source == target:
        return 0
    else:
        return 2


def min_edit_distance(source, target):
    """
    Function to find minimum edit distance between two given strings.

    :param source: The original string to be edited.
    :param target: The string to be edited to.
    :return: The minimum edit distance.
    """
    n = len(source)
    m = len(target)
    d = np.zeros(shape=[n + 1, m + 1])

    # Initialization: the zeroth row and column is the distance from the empty string
    for i in range(1, n + 1):
        d[i, 0] = d[i - 1, 0] + del_cost(source[i - 1])
    for j in range(1, m + 1):
        d[0, j] = d[0, j - 1] + ins_cost(target[j - 1])

    # Recurrence relation:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + del_cost(source[i - 1]),
                          d[i - 1, j - 1] + sub_cost(source[i - 1], target[j - 1]),
                          d[i, j - 1] + ins_cost(target[j - 1]))

    print("Edit Distance Matrix: \n\n{}\n".format(d))

    return d[n, m]


in_p = sys.argv
res = min_edit_distance(str(in_p[1]), str(in_p[2]))
print("Minimum Edit Distance between *{}* and *{}* is {}.".format(str(in_p[1]), str(in_p[2]), res))
