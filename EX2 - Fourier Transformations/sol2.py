import numpy as np
import scipy.io.wavfile as io
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray
from imageio import imread

# ------ ex1 constants ------ #
GRAYSCALE = 1
RGB = 2
ALL_COLORS = 256
NORMALIZATION_FACTOR = 255
BIGGEST_VAL = 255
RGB_DIMS = 3
# --------------------------- #

dx = np.array([[0.5, 0, -0.5]])
dy = np.array([[0.5], [0], [-0.5]])
CHANGE_RATE_FILENAME = "change_rate.wav"
CHANGE_SAMPLE_FILENAME = "change_samples.wav"


# ----------------- ex2_helper ------------------ #
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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


# ------------- ex2 solutions ------------- #

def DFT(signal):
    """
    transforming signal to its fourier form
    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal with shape (N,) or (N,1)
    """
    n = signal.shape[0]
    dft = (np.arange(n)) * (np.arange(n).reshape(-1, 1)).astype(np.complex128)
    dft = np.exp(-2 * dft * np.pi * 1.j / n)
    return (dft.dot(signal.reshape((n,)))).reshape(signal.shape)


def IDFT(fourier_signal):
    """
    inverse the fourier signal back to signal
    :param fourier_signal: n array of dtype complex128 with the
    shape (N,) or (N,1)
    :return: complex signal with shape (N,) or (N,1)
    """
    n = fourier_signal.shape[0]
    idft = (np.arange(n)) * \
           (np.arange(n).reshape(-1, 1)).astype(np.complex128)
    idft = np.exp(2 * idft * np.pi * 1.j / n)
    return (idft.dot(fourier_signal.reshape((n,))) / n).reshape(
        fourier_signal.shape)


def DFT2(image):
    """
    :param image: grayscale image of dtype float64 of shape (M,N) or (M,N,1).
    :return: the transform fourier of the image
    """
    m, n  = image.shape
    image = image.reshape((m, n))
    dft = np.zeros((m, n)).astype(np.complex128)
    for i in range(m):  # fourier transform  of the rows
        dft[i] = DFT(image[i])
    dft = dft.T
    for i in range(n):  # fourier transform of the cols
        dft[i] = DFT(dft[i])
    return dft.T.reshape(shape)


def IDFT2(fourier_image):
    """
    :param fourier_image:  2D array of dtype complex128 of
           shape (M,N) or (M,N,1).
    :return: the inverse transform of the given fourier_image
    """
    shape = fourier_image.shape
    m = fourier_image.shape[0]
    n = fourier_image.shape[1]
    fourier_image = fourier_image.reshape((m, n))
    idft = np.zeros((m, n)).astype(np.complex128)
    for i in range(m):  # fourier transform  of the rows
        idft[i] = IDFT(fourier_image[i])
    idft = idft.T
    for i in range(n):  # fourier transform of the cols
        idft[i] = IDFT(idft[i])
    return idft.T.reshape(shape)


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples,
    but changing the sample rate written in the file header
    :param filename: a string representing the path to a WAV file
    :param ratio:  a positive float64 representing the duration change
    :return: None
    """
    rate, audio = io.read(filename)
    io.write(CHANGE_RATE_FILENAME, int(rate * ratio), audio)


def change_samples(filename, ratio):
    """
    This function will call the function resize to change the number of
    samples by the given ratio
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change.
    :return: 1D ndarray of dtype float64 representing the new sample points
    """
    rate, audio = io.read(filename)
    new_samples = resize(audio, ratio).astype(np.float64)
    io.write(CHANGE_SAMPLE_FILENAME, rate, new_samples)
    return new_samples


def resize(data, ratio):
    """
    :param data: 1D ndarray of dtype float64 or complex128(*)
    representing the original sample points
    :param ratio: value of resize is a 1D ndarray of the dtype of data
    representing the new sample points
    :return: a 1D ndarray of the dtype of data representing the
             new sample points.
    """
    if ratio == 1:
        return data

    n = data.shape[0]
    new_n = int(data.shape[0] / ratio)
    fourier_data = DFT(data)
    fourier_data = np.fft.fftshift(fourier_data)
    fourier_data_resized = np.zeros(new_n).astype(np.complex128)

    if ratio > 1:  # remove samples

        s_remove = n - new_n  # samples amount to remove
        fourier_data_resized = \
            fourier_data[int(s_remove / 2):int(n - (s_remove / 2))]

    if ratio < 1:  # padding with zeros

        z_pad = new_n - n  # zeros amount to add
        fourier_data_resized[int(z_pad / 2):int(new_n - (z_pad / 2))] += \
            fourier_data

    return IDFT(np.fft.ifftshift(fourier_data_resized)).astype(data.dtype)


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram
    scaling. This is done by computing the spectrogram, changing the number
    of spectrogram columns, and creating back the audio.
    :param data: a 1D ndarray of dtype float64 representing the original
                 sample points
    :param ratio: a positive float64 representing the rate change
                  of the WAV file
    :return: new sample points according to ratio with the same
             datatype as data
    """
    spectrogram = stft(data)
    rows, cols = spectrogram.shape
    spectrogram_resized = np.zeros((rows, int(cols / ratio)))
    for i in range(rows):
        spectrogram_resized[i] = resize(spectrogram[i], ratio)
    return istft(spectrogram_resized).astype(data.dtype)


def resize_vocoder(data, ratio):
    """
    Phase vocoding is the process of scaling the spectrogram as done in
    resize_spectrogram, but includes the correction of the phases of each
    frequency according to the shift of each window
    :param data: a 1D ndarray of dtype float64 representing
                 the original sample points
    :param ratio: positive float64 representing the rate change of
                  the WAV file
    :return: the new sample points according to ratio with
             the same datatype as data
    """
    spectrogram = stft(data)
    spectrogram = phase_vocoder(spectrogram, ratio)
    return istft(spectrogram).astype(data.dtype)


def conv_der(im):
    """
    computes the magnitude of image derivatives using convolution
    :param im: grayscale of type float64 image
    :return: the magnitude of the derivative, with the same dtype and shape
    """
    im_dx = signal.convolve2d(im, dx, mode='same')
    im_dy = signal.convolve2d(im, dy, mode='same')

    return np.sqrt(np.abs(im_dx) ** 2 + np.abs(im_dy) ** 2)


def fourier_der(im):
    """
     computes the magnitude of image derivatives using fourier
     :param im:  grayscale of type float64 image
     :return: the magnitude of the derivative, with the same dtype and shape
     """
    rows, cols = im.shape
    fourier_im = np.fft.fftshift(DFT2(im))

    array_ind_x = (np.arange(rows * cols).reshape((rows, cols)) // cols) - \
                  (rows // 2)
    im_dx = (fourier_im * array_ind_x) * 2 * 1.j * np.pi / rows
    im_dx = IDFT2(np.fft.ifftshift(im_dx))

    array_ind_y = np.arange(rows * cols).reshape((rows, cols)) % cols - \
                  (cols // 2)
    im_dy = (fourier_im * array_ind_y) * 2 * 1.j * np.pi / cols
    im_dy = IDFT2(np.fft.ifftshift(im_dy))
    return np.sqrt(np.abs(im_dx) ** 2 + np.abs(im_dy) ** 2)

