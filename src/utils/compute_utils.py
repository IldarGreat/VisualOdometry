import numpy as np


def compute_error(true_state, computed_state, type):
    if true_state.shape != computed_state.shape:
        raise Exception("WWW")
    if type == 'MSE':
        diff = true_state - computed_state
        squared_diff = np.square(diff)
        error = np.mean(squared_diff)
    return error