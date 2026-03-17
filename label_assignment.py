import numpy as np


def argmin_label_map(cost_volume):
    return np.argmin(cost_volume, axis=0).astype(np.uint8)