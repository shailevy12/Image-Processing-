import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
ALL_COLORS = 256
NORMALIZATION_FACTOR = 255
BIGGEST_VAL = 255
RGB_DIMS = 3
GRAYSCALE_DIMS = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    return np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    return np.dot(imYIQ, (np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)).T)


def equalization_helper(im):
    """
    helper function for the histogram_equalize function.
    perforn histogram equalization in the given 2-dimensional image.
    :param im: the 2-dimensional image
    :return: [im after equalization, hist_orig, hist_eq]
    """
    hist_orig = np.histogram(im, bins=ALL_COLORS)[0]
    z = hist_orig.shape[0]  # number of gray levels
    cum = np.cumsum(hist_orig)  # cumulative histogram
    cm = (cum[cum > 0])[0]  # first index content that isn't zero
    T = np.round(((cum - cm) / (cum[z - 1] - cm)) * NORMALIZATION_FACTOR)
    im_eq = T[im.astype(np.int64)]
    hist_eq = np.histogram(im_eq, bins=ALL_COLORS)[0]
    return [im_eq / NORMALIZATION_FACTOR, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    result = []

    if im_orig.ndim == RGB_DIMS:
        im_yiq = rgb2yiq(im_orig)
        result = equalization_helper(im_yiq[:, :, 0] * NORMALIZATION_FACTOR)
        im_yiq[:, :, 0] = result[0]
        result[0] = yiq2rgb(im_yiq)

    if im_orig.ndim == GRAYSCALE_DIMS:
        result = equalization_helper(im_orig * NORMALIZATION_FACTOR)

    return result


def initialization_z(cum_hist, n_quant, s_size):
    """
    initialize the array z for the first time.
    :param cum_hist: commutative histogram
    :param n_quant: number of colors to quantize
    :param s_size: the size of the segment
    :return: the z array
    """
    lst = np.zeros(1)
    for i in range(n_quant - 1):
        lst = np.append(lst, np.where(cum_hist > s_size)[0][0] - 1)
        cum_hist = cum_hist - s_size
    lst = np.append(lst, BIGGEST_VAL)
    return lst.astype(np.int64)


def quantize_process(im, n_quant, n_iter, p):
    """
    helper function for the quantize function.
    :param im: the 2-dimensional image to operate on.
    :param n_quant: number of colors to quantize
    :param n_iter: maximum number of iterations
    :param p: number of pixels of the image
    :return: [map, error] when map is the function which says what pixels
    suppose to be mapped to what pixels.
    """
    hist = np.round(
        np.histogram(im, bins=ALL_COLORS)[0])
    cum_hist = np.cumsum(hist)
    z = initialization_z(cum_hist, n_quant, int(p / n_quant))
    q = np.zeros(n_quant)
    error = np.zeros(1)
    for i in range(n_iter):
        old_z = z.copy()
        for j in range(n_quant):
            if j == 0:  # considering zero as an edge case
                index_array = np.arange(z[j], z[j + 1] + 1)
                sliced_hist = hist[z[j]:z[j + 1] + 1]
            else:
                index_array = np.arange(z[j] + 1, z[j + 1] + 1)
                sliced_hist = hist[z[j] + 1:z[j + 1] + 1]

            q[j] = np.sum(index_array * sliced_hist) / np.sum(sliced_hist)
            error[i] += np.sum(((q[j] - index_array) *
                                (q[j] - index_array)) * sliced_hist)

        for j in range(1, n_quant):
            z[j] = int((q[j - 1] + q[j]) / 2)

        if np.array_equal(z, old_z):
            break

        if i < n_iter - 1:
            error = np.append(error, [0])

    map_f = np.zeros(hist.shape)
    for i in range(n_quant):
        map_f[z[i]:z[i + 1] + 1] = q[i]

    return [map_f, error]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """

    p = im_orig.shape[0] * im_orig.shape[1]  # number of pixels

    if im_orig.ndim == RGB_DIMS:
        im_yiq = rgb2yiq(im_orig)
        result = quantize_process(im_yiq[:, :, 0] * NORMALIZATION_FACTOR,
                                  n_quant, n_iter, p)
        im_yiq[:, :, 0] = result[0][(im_yiq[:, :, 0] *
                                     NORMALIZATION_FACTOR).astype(
            np.int64)] / NORMALIZATION_FACTOR
        im = yiq2rgb(im_yiq)
        return [im, result[1]]

    elif im_orig.ndim == GRAYSCALE_DIMS:
        result = quantize_process(im_orig * NORMALIZATION_FACTOR, n_quant,
                                  n_iter, p)
        im = result[0][(im_orig * NORMALIZATION_FACTOR).astype(
            np.int64)] / NORMALIZATION_FACTOR
        return [im, result[1]]

