import json
import numpy as np

def tiff_to_np(mask):
    height, width = mask.shape
    new_mask = np.zeros((6,) + mask.shape)

    for h in range(height):
        for w in range(width):
            p = int(mask[h, w])
            # convert one one-hot vector
            v = np.zeros((1, 6))
            v[0, p] = 1
            new_mask[:,h, w] = v
    return new_mask


class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)


