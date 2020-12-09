import numpy as np
import math
import cv2
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix
def simplest_cb(img, percent):
    if percent <= 0:
        percent = 5
    img = np.float32(img)
    halfPercent = percent/200.0
    channels = cv2.split(img)
    results = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        flat = np.sort(flat)
        lowVal = flat[int(math.floor(flat.shape[0] * halfPercent))]
        topVal = flat[int(math.ceil(flat.shape[0] * (1.0 - halfPercent)))]
        channel = apply_threshold(channel, lowVal, topVal)
        normalized=cv2.normalize(channel,channel.copy(),0.0,255.0,cv2.NORM_MINMAX)
        channel= np.uint8(normalized)
        results.append(channel)

    return cv2.merge(results)


