# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def mse(e):
    return np.mean(e ** 2)

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    a = tx.T @ tx
    b = tx.T @ y
    
    w = np.linalg.solve(a, b)
    yp = tx @ w
    e = mse(y - yp)

    return w, e
