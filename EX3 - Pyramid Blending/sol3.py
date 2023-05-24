import numpy as np
import os
import scipy.ndimage.filters as f
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
from scipy import signal

# ------ ex1 constants ------ #
GRAYSCALE = 1
RGB = 2
NORMALIZATION_FACTOR = 255
RGB_DIMS = 3

# ------- ex3 constants ------ #
BASE_FILTER = np.array([[1, 1]])
THRESHOLD = 16


# ------------ read_image from ex1 ------------ #
def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    if representation != RGB and representation != GRAYSCALE:
        return

    im = imread(filename).astype(np.float64)

    if im.ndim == RGB_DIMS and representation == GRAYSCALE:
        im = rgb2gray(im)

    return im / NORMALIZATION_FACTOR


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blur_im = f.convolve(im, blur_filter)  # row blurring
    blur_im = f.convolve(blur_im, blur_filter.T)  # col blurring
    return blur_im[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    row, col = im.shape
    zero_padding_im = np.zeros((2 * row, 2 * col))
    zero_padding_im[::2, ::2] = im  # adding zeros in the odd places
    blur_im = f.convolve(zero_padding_im, blur_filter * 2)  # row blurring
    blur_im = f.convolve(blur_im, blur_filter.T * 2)  # col blurring
    return blur_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = [im]
    filter_vec = BASE_FILTER
    for i in range(filter_size - 2):
        filter_vec = signal.convolve2d(filter_vec, BASE_FILTER)
    filter_vec = filter_vec / np.sum(filter_vec)

    for i in range(1, max_levels):
        row, col = pyr[i - 1].shape
        if row <= THRESHOLD or col <= THRESHOLD:
            break
        pyr.append(reduce(pyr[i - 1], filter_vec))

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyr, filter_vec = \
        build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gaussian_pyr) - 1):
        pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter_vec))
    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    n = len(lpyr)
    if n == 1:
        return expand(lpyr[0] * coeff[0], filter_vec)

    if n >= 2:
        img = lpyr[n - 1]
        for i in range(n - 1, 0, -1):
            img = expand(img * coeff[i], filter_vec) + lpyr[i - 1]
        return img


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    for i in range(levels):
        pyr[i] = (pyr[i] - np.min(pyr[i])) / (np.max(pyr[i]) - np.min(pyr[i]))
    size_x, size_y = 0, pyr[0].shape[0]
    for i in range(levels):
        size_x += pyr[i].shape[1]
    res = np.zeros((size_y, size_x))
    k = 0
    for i in range(levels):
        m, n = pyr[i].shape
        res[:m, k:k + n] = pyr[i]
        k += n
    return res


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    l1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    g = build_gaussian_pyramid(mask.astype(np.float64),
                               max_levels, filter_size_mask)[0]
    l_out = []
    for i in range(len(g)):
        l_out.append(g[i] * l1[i] + (1 - g[i]) * l2[i])
    blend_im = laplacian_to_image(l_out, filter_vec, [1] * len(l_out))
    blend_im = np.clip(blend_im, 0, 1)
    return blend_im


def relpath(filename):
    """
    helper function described in the exercise.
    :param filename:
    :return:
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_examples(f1, f2, f3):
    """
    Perform pyramid blending on two images RGB and a mask
    :f1,f2,f3: the file paths to im1, im2 and ,mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im2 = read_image(relpath(f1), RGB)
    im1 = read_image(relpath(f2), RGB)
    mask = read_image(relpath(f3), GRAYSCALE).astype(
        np.int64).astype(np.bool)

    blend_im = np.zeros(im1.shape)
    blend_im[:, :, 0] = \
        pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 7, 3, 3)
    blend_im[:, :, 1] = \
        pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 7, 3, 3)
    blend_im[:, :, 2] = \
        pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 7, 3, 3)

    images = [im1, im2, mask.astype(np.float64), blend_im]
    figure = plt.figure()
    for i in range(len(images)):
        figure.add_subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.axis("off")
    plt.show()

    return im1, im2, mask, blend_im


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    return blending_examples("externals/lobsters.jpg",
                             "externals/rossandrachel.jpg",
                             "externals/mask.jpg")
    # Only who watched friends will appreciate this blending.
    # He's her lobster!!!


def blending_example2():
    """
   Perform pyramid blending on two images RGB and a mask
   :return: image_1, image_2 the input images, mask the mask
       and out the blended image
   """
    return blending_examples("externals/lava.jpg",
                             "externals/waterfall.jpg",
                             "externals/mask2.jpg")
