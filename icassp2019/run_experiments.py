#!/usr/bin/env python3
"""
run_experiments.py: Train primal-dual network on IEEE speech corpus.

Input: specify (*)-(***) below, other files do not need to be modified
"""

# =============================================================================
# python packages
# =============================================================================
import numpy as np
import scipy.io as sio
from pdn_lib import wav_folder_to_matrix, uniform_quantization
from train_pdn import train_pdn

# =============================================================================
# authorship information 
# =============================================================================
__author__ = "Christoph Brauer"
__copyright__ = "Copyright 2018, Christoph Brauer"
__credits__ = ["Christoph Brauer"]

__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christoph Brauer"
__email__ = "ch.brauer@tu-bs.de"
__status__ = "Prototype"

# =============================================================================
# specify parameters for quantization, network architecture and training (*)
# =============================================================================
bitrate = 5 # bitrate for uniform quantization
n = 320 # frame size / number of columns in K
m = 320 # dual dimension / number of rows in K
L = 2 # number of unrolled Chambolle-Pock iterations / primal-dual blocks
E = 3 # number of epochs during training
batch_size = 320 # will be ignored in case weighting filter objective is used
learning_rate = 1e-4
dct_init = True # K ist initialized with DCT (True) or randomly (False)
tau = 0 # initial value for step size parameter
sigma = 10 # initial value for step size parameter
folder = '' # results will be written to this folder

# =============================================================================
# import and quantize speech signals (**)
# =============================================================================
if not 'S_train' in locals():
    # import speech signal snippets
    STilde_train, indices_train = wav_folder_to_matrix('IEEE_corpus/train_data/', n)
    STilde_dev, indices_dev = wav_folder_to_matrix('IEEE_corpus/dev_data/', n)
    print('finished data import...')
    # quantize speech signal snippets
    S_train = uniform_quantization(STilde_train, bitrate)
    S_dev = uniform_quantization(STilde_dev, bitrate)
    print('finished quantization...')
    # import filters for objective function
    H_train = np.column_stack((sio.loadmat('IEEE_corpus/train_data_filters_pt1.mat')['H1'],
                               sio.loadmat('IEEE_corpus/train_data_filters_pt2.mat')['H2']))
    H_dev = sio.loadmat('IEEE_corpus/dev_data_filters.mat')['H']
    # flip weighting filters
    H_train = np.flip(H_train, 0)
    H_dev = np.flip(H_dev, 0)
    print('finished filter import...')

# =============================================================================
# start training
# (pass H_train=None instead of H_train=H_train if standard MSE loss shall be
# used instead of weighting filter based loss) (***)
# =============================================================================
Delta = 2 / np.power(2, bitrate) # quantization interval length
train_pdn(n, m, L, E, Delta, S_train, S_dev, STilde_train, STilde_dev,
          dct_init=dct_init, tau=tau, sigma=sigma,
          learning_rate=learning_rate, batch_size=batch_size,
          H_train=H_train, H_dev=H_dev, indices_train=indices_train,
          indices_dev=indices_dev, folder=folder)