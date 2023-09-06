"""
auxiliary_functions.py

This file contains auxiliary functions for unrolling of the primal-dual algorithm.
"""

# =============================================================================
# python packages
# =============================================================================
import numpy as np
import os
import tensorflow as tf
from scipy.io import wavfile

# =============================================================================
# authorship information
# =============================================================================
__author__ = 'Christoph Brauer'
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Christoph Brauer'
__email__ = 'christoph.brauer@dlr.de'
__status__ = 'Prototype'


# =============================================================================
# functions for data import
# =============================================================================

def decode(serialized_example):
    """ Function to decode serialized examples from TensorFlow Dataset format

    :param serialized_example: serialized input-output pair
    :return: tuple of two nx1 vectors / decoded input-output pair
    """

    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'y': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    x = tf.io.parse_tensor(parsed_example['x'], tf.float32)
    y = tf.io.parse_tensor(parsed_example['y'], tf.float32)
    return x, y


def get_data(bitrate=3, batch_size=32, repeat_data=None, shuffle_data=True):
    """ Function to load IEEE speech corpus data from TensorFlow Dataset format

    :param bitrate: positive integer / paper uses 4
    :param batch_size: positive integer / batch size during training
    :param repeat_data: positive integer or None / TensorFlow Dataset specific parameter
    :param shuffle_data: boolean / whether to shuffle data in data loading pipeline
    :return: training, validation and test data in TensorFlow Dataset format, and number of training examples
    """

    file_train = f'IEEE_corpus/ieee_speech_train_{bitrate}bit.tfrecord',
    file_val = f'IEEE_corpus/ieee_speech_val_{bitrate}bit.tfrecord',
    file_test = f'IEEE_corpus/ieee_speech_test_{bitrate}bit.tfrecord',

    ds_tr = tf.data.TFRecordDataset([file_train])
    ds_tr = ds_tr.map(decode)
    num_tr_ex = 0
    for _ in ds_tr:
        num_tr_ex += 1
    ds_tr = ds_tr.repeat(repeat_data)
    if shuffle_data:
        ds_tr = ds_tr.shuffle(1000, reshuffle_each_iteration=True)
    ds_tr = ds_tr.batch(batch_size)
    ds_tr = ds_tr.prefetch(tf.data.experimental.AUTOTUNE)

    ds_v = tf.data.TFRecordDataset([file_val])
    ds_v = ds_v.map(decode)
    ds_v = ds_v.batch(batch_size)
    ds_v = ds_v.prefetch(tf.data.experimental.AUTOTUNE)

    ds_te = tf.data.TFRecordDataset([file_test])
    ds_te = ds_te.map(decode)
    ds_te = ds_te.batch(batch_size)
    ds_te = ds_te.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_tr, ds_v, ds_te, num_tr_ex


# =============================================================================
# functions to run primal-dual algorithm in different contexts
# =============================================================================

def chambolle_pock(K, x, eta, L, tau, sigma, theta):
    """Runs the primal-dual algorithm (Algorithm 1) on problem (14) and returns primal iterate after n_iter iterations

    :param K: mxn matrix / analysis operator
    :param x: nx1 vector / observation
    :param eta: positive scalar / rhs-parameter depending on bitrate
    :param L: positive integer / number of iterations
    :param tau: positive scalar / dual step size (ensure sigma * tau * ||K||**2 < 1)
    :param sigma: positive scalar / primal step size (ensure sigma * tau * ||K||**2 < 1)
    :param theta: scalar in [0, 1] / extrapolation factor
    :return: nx1 vector / approximate solution after n_iter iterations
    """

    # initialize forward sequence (current iterates, and in primal case also previous for extrapolation)
    y_curr = x
    y_prev = x
    psi_curr = tf.zeros((x.shape[0], tf.shape(K)[0]))
    # iterate forwards
    for _ in range(L):
        # extrapolation
        y_bar = y_curr + theta * (y_curr - y_prev)
        # dual update
        z_d = psi_curr + sigma * y_bar @ tf.transpose(K)
        psi_curr = tf.clip_by_value(z_d, -1, 1)
        # primal update
        y_prev = y_curr
        z_p = y_curr - x - tau * psi_curr @ K
        y_curr = tf.clip_by_value(z_p, -eta, eta) + x
    return y_curr


def full_bp(K, x, eta, L, tau, sigma, theta, y, b, eta_backward):
    """Runs the primal-dual algorithm on problem (14) for L forward iterations (Algorithm 1) and performs
    b backward iterations (Algorithm 2) for gradient computation. Usage with b = L computes
    the full unrolling gradient and is referred to as FULL BP in the Numerical Experiments section.

    :param K: mxn matrix / analysis operator
    :param x: nx1 vector / observation
    :param eta: positive scalar / rhs-parameter depending on bitrate
    :param L: positive integer / number of iterations
    :param tau: positive scalar / dual step size (ensure sigma * tau * ||K||**2 < 1)
    :param sigma: positive scalar / primal step size (ensure sigma * tau * ||K||**2 < 1)
    :param theta: scalar in [0, 1] / extrapolation factor
    :param y: nx1 vector / ground truth (optimal solution)
    :param b: positive integer / number of backward iterations
    :param eta_backward: positive scalar / eta to be used in backward passed (eventually > eta to increase stability)
    :return: gradient and primal iterate after L iterations
    """

    # initialize forward sequence (current iterates, and in primal case also previous for extrapolation)
    y_curr = x
    y_prev = x
    psi_curr = tf.zeros((x.shape[0], tf.shape(K)[0]))
    y_bars = []
    psis = []
    z_ds = []
    z_ps = []
    # iterate forwards
    for i in range(L):
        # extrapolation
        y_bar = y_curr + theta * (y_curr - y_prev)
        # dual update
        z_d = psi_curr + sigma * y_bar @ tf.transpose(K)
        psi_curr = tf.clip_by_value(z_d, -1, 1)
        # primal update
        y_prev = y_curr
        z_p = y_curr - x - tau * psi_curr @ K
        y_curr = tf.clip_by_value(z_p, -eta, eta) + x
        if L - 1 - i < b:
            y_bars.append(y_bar)
            psis.append(psi_curr)
            z_ds.append(z_d)
            z_ps.append(z_p)
    # initialize backward sequence
    delta_d = tf.zeros(tf.shape(psi_curr))
    delta_d_prev = delta_d
    delta_p = y_curr - y
    grad_K = tf.zeros(tf.shape(K))
    # iterate backwards
    for i in range(min(L, b)):
        delta_d_bar = delta_d + theta * (delta_d - delta_d_prev)
        delta_d_prev = delta_d
        mask_p = tf.cast(tf.abs(z_ps[-1 - i] - x) <= eta_backward, tf.float32)
        mask_d = tf.cast(tf.abs(z_ds[-1 - i]) <= 1, tf.float32)
        delta_p = mask_p * (delta_p + sigma * delta_d_bar @ K)
        delta_d = mask_d * (delta_d - tau * delta_p @ tf.transpose(K))
        grad_K += sigma * tf.transpose(delta_d) @ y_bars[-1 - i] - tau * tf.transpose(psis[-1 - i]) @ delta_p
    return grad_K, y_curr


def truncated_bp(K, x, eta, L, tau, sigma, theta, y, b, eta_backward):
    """Runs the primal-dual algorithm on problem (14) for L forward iterations (Algorithm 1) and performs
    b backward iterations for gradient computation. This function uses formula (12) with primal and dual
    optimal solutions replaced with respective iterates after L iterations. Also, to compute Delta_P and
    Delta_D in (12), only the resulting z_P and z_D after L iterations are used, and both series are approximated
    with b terms.

    :param K: mxn matrix / analysis operator
    :param x: nx1 vector / observation
    :param eta: positive scalar / rhs-parameter depending on bitrate
    :param L: positive integer / number of iterations
    :param tau: positive scalar / dual step size (ensure sigma * tau * ||K||**2 < 1)
    :param sigma: positive scalar / primal step size (ensure sigma * tau * ||K||**2 < 1)
    :param theta: scalar in [0, 1] / extrapolation factor
    :param y: nx1 vector / ground truth (optimal solution)
    :param b: positive integer / number of backward iterations
    :param eta_backward: positive scalar / eta to be used in backward passed (eventually > eta to increase stability)
    :return: gradient and primal iterate after L iterations
    """

    # initialize forward sequence (current iterates, and in primal case also previous for extrapolation)
    y_curr = x
    y_prev = x
    psi_curr = tf.zeros((x.shape[0], tf.shape(K)[0]))
    # iterate forwards
    for i in range(L):
        # extrapolation
        y_bar = y_curr + theta * (y_curr - y_prev)
        # dual update
        z_d = psi_curr + sigma * y_bar @ tf.transpose(K)
        psi_curr = tf.clip_by_value(z_d, -1, 1)
        # primal update
        y_prev = y_curr
        z_p = y_curr - x - tau * psi_curr @ K
        y_curr = tf.clip_by_value(z_p, -eta, eta) + x
    # initialize backward sequence
    delta_d = tf.zeros(tf.shape(psi_curr))
    delta_d_prev = delta_d
    delta_p = y_curr - y
    grad_K = tf.zeros(tf.shape(K))
    mask_p = tf.cast(tf.abs(z_p - x) <= eta_backward, tf.float32)
    mask_d = tf.cast(tf.abs(z_d) <= 1, tf.float32)
    # iterate backwards
    for i in range(min(L, b)):
        delta_d_bar = delta_d + theta * (delta_d - delta_d_prev)
        delta_d_prev = delta_d
        delta_p = mask_p * (delta_p + sigma * delta_d_bar @ K)
        delta_d = mask_d * (delta_d - tau * delta_p @ tf.transpose(K))
        grad_K += sigma * tf.transpose(delta_d) @ y_curr - tau * tf.transpose(psi_curr) @ delta_p
    return grad_K, y_curr


def chambolle_pock_distances(K, x, eta, L, tau, sigma, theta, y):
    """Runs the primal-dual algorithm (Algorithm 1) on problem (14) and returns distances of first L iterates to
    ground truth y

    :param K: mxn matrix / analysis operator
    :param x: nx1 vector / observation
    :param eta: positive scalar / rhs-parameter depending on bitrate
    :param L: positive integer / number of iterations
    :param tau: positive scalar / dual step size (ensure sigma * tau * ||K||**2 < 1)
    :param sigma: positive scalar / primal step size (ensure sigma * tau * ||K||**2 < 1)
    :param theta: scalar in [0, 1] / extrapolation factor
    :param y: nx1 vector / ground truth (optimal solution)
    :return: list / distances of iterates from optimal solution
    """

    # initialize list for distances
    distances = []
    # initialize forward sequence (current iterates, and in primal case also previous for extrapolation)
    y_curr = x
    y_prev = x
    psi_curr = tf.zeros((x.shape[0], tf.shape(K)[0]))
    distances.append(tf.reduce_mean((y_curr - y) ** 2))
    # iterate forwards
    for _ in range(L):
        # extrapolation
        y_bar = y_curr + theta * (y_curr - y_prev)
        # dual update
        z_d = psi_curr + sigma * y_bar @ tf.transpose(K)
        psi_curr = tf.clip_by_value(z_d, -1, 1)
        # primal update
        y_prev = y_curr
        z_p = y_curr - x - tau * psi_curr @ K
        y_curr = tf.clip_by_value(z_p, -eta, eta) + x
        distances.append(tf.reduce_mean((y_curr - y) ** 2))
    return distances


# =============================================================================
# functions to encode ieee speech corpus data in TensorFlow Dataset format
# =============================================================================

def wav_to_matrix(filename, n):
    """ Create frames from single .wav file

    :param filename: string / path to a single .wav file
    :param n: positive integer / frame length
    :return: nx(number of frames) matrix
    """
    # read .wav file
    _, y = wavfile.read(filename)
    # normalize signal
    y = y / (np.max(np.abs(y)) + 1e-10)
    # reshape signal
    if np.mod(y.shape[0], n) > 0:
        y_reshape = np.reshape(y[:-np.mod(y.shape[0], n)], (n, -1), order='F')
    else:
        y_reshape = np.reshape(y, (n, -1), order='F')
    return y_reshape


def wav_folder_to_matrix(folder, n):
    """ Create frames from folder of .wav files

    :param folder: string / path to folder with .wav files
    :param n: positive integer / frame length
    :return: nx(number of frames) matrix
    """
    # initialize signal matrix
    Y = np.empty((n, 0))
    # get names of files in folder
    filenames = sorted(os.listdir(folder))
    # apply wav_to_matrix to each file in folder
    for filename in filenames:
        y_reshape = wav_to_matrix(folder + '/' + filename, n)
        Y = np.column_stack((Y, y_reshape))
    return Y


def uniform_quantization(Y, bitrate):
    """ Apply uniform quantization to speech data

    :param Y: matrix / speech data frames
    :param bitrate: quantization bitrate
    :return: matrix (same size as Y) / quantized speech data frames
    """
    # compute length of quantization intervals
    Delta = 2 / np.power(2, bitrate)
    # quantize signal matrix
    X = np.multiply(np.sign((Y >= 0) * 2 - 1), Delta * (np.floor(np.abs(Y) / Delta) + .5))
    return X
