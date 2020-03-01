import numpy as np
from skimage.segmentation import felzenszwalb
from skimage import data
from skimage.color import rgb2yiq
import collections
from scipy import signal
import scipy as sc
import math as m

def gaussian_kernel(k, s = 0.5):
    # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    probs = [m.exp(-z*z/(2*s*s))/m.sqrt(2*m.pi*s*s) for z in range(-k,k+1)]
    return np.outer(probs, probs)


def find_all_neigbors(x, y, height, width):
    neighbors = []
    num_vertice = 0
    for i in range(height):
        for j in range(width):
            if (i != x or j != y) and np.sqrt((np.power(i-x, 2)+np.power(j-y, 2))) <= 1:
                neighbors.append([i, j, num_vertice])
            num_vertice += 1
    return neighbors


def graph_based_segmentation(image, scale, sigma, min_size, sigma_x):
    """
    Felzenszwalb segmentation
    :param image: a numpy array represents image with channel
    :param scale:
    :param sigma:
    :param min_size: min size for a region
    :return: a list of regions
    [0 0 0 0]
    [0 0 0 0]
    [0 0 0 0]
    [0 0 0 0]


    """
    matrix = np.random.randint(255, size=(4, 4, 3))
    height, width, deep = matrix.shape
    adj_matrix = np.zeros((height*width, height*width))
    num_vertice = 0
    degree_matrix = np.zeros((height*width, height*width))
    seg_class = [i for i in range(height*width)]
    for i in range(height):
        for j in range(width):
            sum_degree = 0
            for neighbor in find_all_neigbors(i, j, height, width):
                adj_matrix[num_vertice, neighbor[2]] = np.exp(
                    -(m.fabs(np.sum(matrix[neighbor[0], neighbor[1], :]-matrix[i, j, :]))/sigma)
                    -(np.sqrt((np.power(neighbor[0]-i, 2)+np.power(neighbor[1]-j, 2)))/sigma_x))
                adj_matrix[neighbor[2], num_vertice] = np.exp(
                    -(m.fabs(np.sum(matrix[neighbor[0], neighbor[1], :] - matrix[i, j, :])) / sigma)
                    - (np.sqrt((np.power(neighbor[0] - i, 2) + np.power(neighbor[1] - j, 2))) / sigma_x))
                sum_degree += adj_matrix[num_vertice, neighbor[2]]
            degree_matrix[num_vertice, num_vertice] = sum_degree
            num_vertice += 1
    # apply paper research paper to create seg classes
    # print(np.sort(np.unique(adj_matrix.flatten())))
    # print(sc.sparse.csgraph.minimum_spanning_tree(adj_matrix))
    return adj_matrix, degree_matrix


def compute_euclidian_dist(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))


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
    # we can reduce compute cost by 2, because similarity matrix is symetric
    similarity_regions = np.zeros((np.max(regions), np.max(regions)))
    for i in range(np.max(regions)):
        for j in range(np.max(regions)):
            if i%25==0:
                print("Index i: {}, j: {}".format(i, j))
            similarity_regions[i, j] = compute_color_similarity(image, regions, i, j)
    print(similarity_regions)


if __name__ == '__main__':
    graph_based_segmentation("aaa", "", 255, "", 1)