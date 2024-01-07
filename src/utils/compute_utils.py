import numpy as np


def compute_error(true_state, computed_state, type):
    if type == 'MSE':
        diff = np.diff([true_state, computed_state])
        squared_diff = np.square(diff)
        error = np.mean(squared_diff) / 1e3
    return error