# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def mae(e):
    return np.mean(np.abs(e))

def mse(e):
    return np.mean(e ** 2)

def compute_loss(y, tx, w, L=mse):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    
    return L(e)
