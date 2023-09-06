"""
run_evaluation.py: Evaluate unrolled primal-dual algorithm on IEEE speech corpus to
reproduce results from the EUSIPCO 2023 paper "Asymptotic analysis and truncated
backpropagation for the unrolled primal-dual algorithm"

Input: Be sure that the parameters under (*) and (**) match the ones you potentially
modified in run_training.py, and that the outputs of run_training.py are in the
folder logs.
"""

# =============================================================================
# import python packages
# =============================================================================
import numpy as np
import scipy.fftpack as fftpack
import tensorflow as tf

# =============================================================================
# import own code
# =============================================================================
from auxiliary_functions import chambolle_pock_distances

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
# specify hyperparameter values to be considered below (*)
# =============================================================================
L_values = [10, 20, 30, 40, 60, 80, 100, 200, 500, 1000]
diff_modes = ['truncated_bp', 'full_bp']


# =============================================================================
# specify other parameters that will be kept constant over all below runs (**)
# =============================================================================
bitrate = 4
sigma_tau_weighting = 10
theta = 1

# =============================================================================
# load test data
# =============================================================================
X_test = np.loadtxt('IEEE_corpus/inputs_test.txt')
Y_test = np.loadtxt('IEEE_corpus/outputs_test.txt')

# =============================================================================
# perform evaluation for different hyperparameter configurations
# =============================================================================
eta = 1 / (2 ** bitrate)

for L in L_values:

    log_dir = f'logs/{L}_iter'

    for diff_mode in diff_modes:

        specs_string = f'{diff_mode}_{L}'
        print(specs_string)

        # load K
        K = tf.cast(np.loadtxt(log_dir + f'/kernel_{specs_string}.txt'), tf.float32)
        norm = tf.linalg.norm(K, ord=2)
        sigma = sigma_tau_weighting * .99 / norm
        tau = 1 / sigma_tau_weighting * .99 / norm

        # evaluate and save results
        distances = chambolle_pock_distances(K, X_test, eta, 100000, tau, sigma, theta, Y_test)
        np.savetxt(log_dir + f'/distances_{specs_string}.txt', np.array(distances))

# do the same with dct K
K = tf.constant(tf.cast(fftpack.dct(np.identity(320), axis=0, type=2, norm='ortho'), tf.float32))
norm = tf.linalg.norm(K, ord=2)
sigma = sigma_tau_weighting * .99 / norm
tau = 1 / sigma_tau_weighting * .99 / norm
distances = chambolle_pock_distances(K, X_test, eta, 100000, tau, sigma, theta, Y_test)
np.savetxt(f'logs/dct/distances.txt', np.array(distances))
