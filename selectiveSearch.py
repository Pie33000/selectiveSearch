import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage import data
import matplotlib.pyplot as plt
import collections


def compute_color_similarity(image, regions, region_i, region_j):
    """
    Compute color similarity between two regions
    :param image: a matrix image shape m*n*c
    :param regions: matrix region computed from image shape m*n
    :param region_i: index of 1st region
    :param region_j: index of 2nd region
    :return: a similarity score, which is a float in interval [0, 1]
    """
    condition_i = regions == region_i
    condition_j = regions == region_j
    score = 0
    for i in range(image.shape[2]):
        counters_i = dict(collections.Counter(np.extract(condition_i, image[:, :, i])).most_common(25))
        counters_j = dict(collections.Counter(np.extract(condition_j, image[:, :, i])).most_common(25))
        factor_i = 1.0 / sum(counters_i.values())
        factor_j = 1.0 / sum(counters_i.values())
        counters_i_normalized = {k: v * factor_i for k, v in counters_i.items()}
        counters_j_normalized = {k: v * factor_j for k, v in counters_j.items()}
        s_i = set(counters_i_normalized)
        s_j = set(counters_j_normalized)
        intersection = s_i & s_j
        for k in list(intersection):
            score += min(counters_i_normalized[k], counters_j_normalized[k])
    return score/image.shape[2]


def compute_texture_similarity(image, regions, region_i, region_j):
    """
    Compute texture similarity between two regions
    :param image: a matrix image shape m*n*c
    :param regions: matrix region computed from image shape m*n
    :param region_i: index of 1st region
    :param region_j: index of 2nd region
    :return: a similarity score, which is a float in interval [0, 1]
    """
    return


def hierarchical_grouping():
    """
    Grouping image to have only one region in output og algorithm
    :return: set of object location hypotheses L (which is a matrix of image segment)
    """
    image = data.astronaut()
    regions = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    print(compute_color_similarity(image, regions, 1, 3))


if __name__ == '__main__':
    hierarchical_grouping()