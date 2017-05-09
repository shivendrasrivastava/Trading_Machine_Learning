"""
Ningmu Zou
nzou3@gatech.edu
mc3h1_defeat_learners
"""

import numpy as np
import math


# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.normal(size=(500, 2))
    Y = X[:, 0] + X[:, 1]
    return X, Y


def best4RT(seed=1489683273):
    np.random.seed(seed)
    X = np.empty((500, 2))
    X[:, 0] = np.random.randint(0, 100, 500)
    X[:, 1] = np.random.randint(0, 100, 500)
    Y = X[:, 0] ** 6 + X[:, 1] ** 6

    return X, Y


if __name__ == "__main__":
    print "they call me Tim."
