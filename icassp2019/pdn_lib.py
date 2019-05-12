#!/usr/bin/env python3

"""
pdn_lib.py: Helper function library for primal-dual networks.

This file provides several auxiliary functions used in deep_pd_tied.py.
"""

# =============================================================================
# python packages
# =============================================================================
from scipy.io import wavfile
import scipy.fftpack as fftpack
import tensorflow as tf
import numpy as np
import os

# =============================================================================
# authorship information 
# =============================================================================
__author__ = "Christoph Brauer"
__copyright__ = "Copyright 2019, Christoph Brauer"
__credits__ = ["Christoph Brauer"]

__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christoph Brauer"
__email__ = "ch.brauer@tu-bs.de"
__status__ = "Prototype"

# =============================================================================
# functions for data import
# =============================================================================
def wav_to_matrix(filename, n):
    # read .wav file
    _, sTilde = wavfile.read(filename)
    # normalize signal
    sTilde = sTilde / (np.max(np.abs(sTilde)) + 1e-10)
    # split signal into frames
    if np.mod(sTilde.shape[0], n) > 0:
        STilde = np.reshape(sTilde[:-np.mod(sTilde.shape[0], n)], (n, -1), order='F')
    else:
        STilde = np.reshape(sTilde, (n, -1), order='F')
    return STilde

def wav_folder_to_matrix(folder, n):
    # initialize signal matrix
    STilde = np.empty((n, 0))
    # get names of files in folder
    filenames = sorted(os.listdir(folder))
    num_frames = np.zeros(len(filenames) + 1)
    # apply wav_to_matrix to each file in folder
    i = 1
    for filename in filenames:
        STilde_tmp = wav_to_matrix(folder + '/' + filename, n)
        STilde = np.column_stack((STilde, STilde_tmp))
        num_frames[i] = int(STilde.shape[1])
        i = i + 1
    return STilde, num_frames.astype(int)

# =============================================================================
# batch generator for stochastic gradient descent
# =============================================================================
def create_batch_generator(S, STilde, batch_size, shuffle=False):
    # set up copies of S and STilde
    S_copy = np.array(S)
    STilde_copy = np.array(STilde)
    # shuffle columns of S and Y uniformly
    if shuffle:
        data = np.row_stack((S_copy, STilde_copy))
        np.random.shuffle(np.transpose(data))
        S_copy = data[:S.shape[0], :]
        STilde_copy = data[S.shape[0]:S.shape[0]+STilde.shape[0], :]
    # generate minibatches
    for i in range(0, S.shape[1], batch_size):
        yield (S_copy[:, i:i+batch_size], STilde_copy[:, i:i+batch_size])

# =============================================================================
# functions for data manipulation in numpy
# =============================================================================
def uniform_quantization(STilde, bitrate):
    # compute length of quantization intervals
    Delta = 2 / np.power(2, bitrate)
    # quantize signal matrix
    S = np.multiply(np.sign((STilde >= 0) * 2 - 1),
                    Delta * (np.floor(np.abs(STilde) / Delta) + .5))
    return S

# =============================================================================
# functions for data manipulation in tensorflow
# =============================================================================
def construct_pdn(n, m, L, Delta, dct_init=False, tau=None, sigma=None,
                         learning_rate=1e-4, filter_use=False):
    # define tensor flow graph
    g = tf.Graph()
    with g.as_default():
        # placeholder for input signals
        S = tf.placeholder(dtype=tf.float64, shape=(n, None))
        # placeholder for target signals
        STilde = tf.placeholder(dtype=tf.float64, shape=(n, None))
        # placeholder for filters
        if filter_use:
            H = tf.placeholder(dtype=tf.float64, shape=(n, None))
        else:
            H = None
        # variables for weights        
        weights = {}
        if dct_init:
            weights['K'] = tf.get_variable('K', dtype=tf.float64,
                   initializer=fftpack.dct(np.identity(n), axis=0, type=2, norm='ortho'))
        else:
            weights['K'] = tf.get_variable('K', shape=(m, n), dtype=tf.float64)
        if tau is not None:
            weights['tau'] = tf.get_variable('tau', initializer=np.float64(tau), dtype=tf.float64 )
        else:
            weights['tau'] = tf.get_variable('tau', shape=(1), dtype=tf.float64)
        if sigma is not None:
            weights['sigma'] = tf.get_variable('sigma', initializer=np.float64(sigma), dtype=tf.float64)
        else:
            weights['sigma'] = tf.get_variable('sigma', shape=(1), dtype=tf.float64)            
        # forward mapping
        dual_act = {}
        prim_act = {}
        dual_act['y1'] = linf_proj_tf(weights['sigma'] * tf.matmul(weights['K'], S), 1)
        prim_act['r1'] = linf_proj_tf(-weights['tau'] * tf.matmul(
                tf.transpose(weights['K']), dual_act['y1']), Delta / 2)
        for l in range(2, L+1):
            prev_lay = '%d' % (l - 1)
            curr_lay = '%d' % (l)
            dual_act['y' + curr_lay] = linf_proj_tf(tf.add(dual_act['y' + prev_lay],
                    weights['sigma'] * tf.matmul(weights['K'], tf.add(prim_act['r' + prev_lay], S))), 1)
            prim_act['r' + curr_lay] = linf_proj_tf(tf.add(prim_act['r' + prev_lay],
                    -weights['tau'] * tf.matmul(tf.transpose(weights['K']), dual_act['y' + curr_lay])), Delta / 2)
        sHat = tf.add(prim_act['r' + '%d' % (L)], S)
        # loss function
        if filter_use:
            paddings_1 = tf.constant([[0, 0], [1, 0]])
            paddings_2 = tf.constant([[0, 319], [0, 0]])
            sTmp = tf.concat([tf.pad(STilde[1:, :-1], paddings_1, 'CONSTANT'), STilde], 0)
            sPad = tf.pad(sTmp, paddings_2, 'CONSTANT')
            f4D = tf.reshape(sPad, [1, 1, 958, -1])
            sHatTmp = tf.concat([tf.pad(sHat[1:, :-1], paddings_1, 'CONSTANT'), sHat], 0)
            sHatPad = tf.pad(sHatTmp, paddings_2, 'CONSTANT')
            fHat4D = tf.reshape(sHatPad, [1, 1, 958, -1])
            H4D = tf.reshape(H, [1, 320, -1, 1])
            conv2D = tf.nn.depthwise_conv2d(f4D - fHat4D, H4D, strides=[1, 1, 1, 1] , padding='VALID')
            conv2DOverlap = tf.contrib.signal.overlap_and_add(tf.transpose(conv2D[0, 0, :, :]), 320)
            loss = tf.nn.l2_loss(conv2DOverlap)
        else:
            loss = tf.losses.mean_squared_error(STilde, sHat)
        # optimization algorithm
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # training operation
        train = optim.minimize(loss)
        return g, S, STilde, H, weights, loss, train
    
def linf_proj_tf(x, r):
    proj = tf.minimum(np.float64(r), tf.maximum(np.float64(-r), x))
    return proj