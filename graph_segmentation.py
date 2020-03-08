import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from filter import *


class universe:
    def __init__(self, n_elements):
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # p

    def size(self, x):
        return self.elts[x, 1]

    def num_sets(self):
        return self.num

    def find(self, x):
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x, 2] = y
        return y

    def join(self, x, y):
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]
        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num -= 1


def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = np.random.randint(0, 255)
    rgb[1] = np.random.randint(0, 255)
    rgb[2] = np.random.randint(0, 255)
    return rgb


def compute_dissimilarity(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))


def get_threshold(size, c):
    return c / size


def segment_graph(num_vertices, num_edges, edges, c):
    edges = edges[edges[:, 2].argsort()]
    u = universe(num_vertices)
    threshold = np.zeros(shape=num_vertices, dtype=float)

    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)

    for i in range(num_edges):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])

        if a != b:
            if (edges[i, 2] <= threshold[a]) and (edges[i, 2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = edges[i, 2] + get_threshold(u.size(a), c)
    return u


def create_graph(in_image, sigma, k, min_size):
    height, width, band = in_image.shape
    in_image_smooth = np.zeros((height, width, band))
    in_image_smooth[:, :, 0] = smooth(in_image[:, :, 0], sigma)
    in_image_smooth[:, :, 1] = smooth(in_image[:, :, 1], sigma)
    in_image_smooth[:, :, 2] = smooth(in_image[:, :, 2], sigma)

    if band != 3:
        raise Exception("This library accept only RGB images at this time")

    edges_size = width * height * 4 - (4*2) - ((height-2)*2) - ((width-2)*2)
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            node = int(y * width + x)
            if y > 0:
                edges[num, 0] = node
                edges[num, 1] = int((y-1)*width+x)
                edges[num, 2] = compute_dissimilarity(in_image_smooth[y, x], in_image_smooth[y-1, x])
                num += 1
            if y < height-1:
                edges[num, 0] = node
                edges[num, 1] = int((y+1)*width+x)
                edges[num, 2] = compute_dissimilarity(in_image_smooth[y, x], in_image_smooth[y+1, x])
                num += 1
            if x < width-2:
                edges[num, 0] = node
                edges[num, 1] = int(y*width+(x+1))
                edges[num, 2] = compute_dissimilarity(in_image_smooth[y, x], in_image_smooth[y, x+1])
                num += 1
            if x > 0:
                edges[num, 0] = node
                edges[num, 1] = int(y*width+(x-1))
                edges[num, 2] = compute_dissimilarity(in_image_smooth[y, x], in_image_smooth[y, x-1])
                num += 1
    u = segment_graph(width*height, num, edges, k)
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    output = np.zeros(shape=(height, width, 3))

    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x, :] = colors[comp, :]

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    output = output.astype(int)
    plt.imshow(output)
    a.set_title('Segmented Image')
    plt.show()


if __name__ == '__main__':
    sigma = 0.9
    k = 500
    min = 20
    in_image = imread("paris.jpg")
    create_graph(in_image, sigma, k, min)