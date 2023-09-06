"""
run_training.py: Train unrolled primal-dual algorithm on IEEE speech corpus to
reproduce results from the EUSIPCO 2023 paper "Asymptotic analysis and truncated
backpropagation for the unrolled primal-dual algorithm"

Input: You can make modifications under (*) and (**) below. Keep the values as
they are to reproduce the results from the paper. Be sure that the data are in
the folder IEEE_corpus. Other inputs are not needed.
"""

# =============================================================================
# import python packages
# =============================================================================
import numpy as np
import os
import scipy.fftpack as fftpack
import tensorflow as tf
import time
from tensorflow import keras

# =============================================================================
# import own code
# =============================================================================
from auxiliary_functions import full_bp, truncated_bp, chambolle_pock, get_data

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
m = 320
bitrate = 4
batch_size = 32
epochs = 20
initial_learning_rate = 1e-5
final_learning_rate = 1e-8
sigma_tau_weighting = 10
theta = 1
eta_backward = 1

# =============================================================================
# load training and validation data
# =============================================================================
ds_tr, _, _, num_tr_ex = get_data(bitrate=bitrate, batch_size=batch_size, repeat_data=1)
X_val = np.loadtxt('IEEE_corpus/inputs_val.txt')
Y_val = np.loadtxt('IEEE_corpus/outputs_val.txt')

# =============================================================================
# train with different hyperparameter configurations
# =============================================================================
eta = 1 / (2 ** bitrate)
os.mkdir('logs')

for L in L_values:

    log_dir = f'logs/{L}_iter'
    os.mkdir(log_dir)

    for diff_mode in diff_modes:

        if diff_mode == 'full_bp':
            grad_fun = full_bp
            b = L
        elif diff_mode == 'truncated_bp':
            grad_fun = truncated_bp
            b = 30

        specs_string = f'{diff_mode}_{L}'
        print(specs_string)

        # set learning rate schedule and optimizer
        schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, epochs * num_tr_ex,
                                                              end_learning_rate=final_learning_rate, power=1.0)
        optimizer = keras.optimizers.Adam(learning_rate=schedule)

        # initialize params
        K = tf.Variable(tf.cast(fftpack.dct(np.identity(320), axis=0, type=2, norm='ortho')[:m, :], tf.float32))

        # initialize training loss
        train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)

        # define train step function
        @tf.function
        def train_step(x, y):
            # compute K norm
            norm = tf.linalg.norm(K, ord=2)
            sigma = sigma_tau_weighting * .99 / norm
            tau = 1 / sigma_tau_weighting * .99 / norm
            # perform optimization steps before gradient computation
            grad, pred = grad_fun(K, x, eta, L, tau, sigma, theta, y, b, eta_backward)
            grads = [grad]
            # apply gradients
            optimizer.apply_gradients(zip(grads, [K]))
            # update metric
            train_loss(tf.reduce_mean((pred - y) ** 2))

        # initialize arrays for learning curves
        loss_values = []
        loss_values_val = []
        execution_times = []
        # append validation loss
        norm = tf.linalg.norm(K, ord=2)
        sigma = sigma_tau_weighting * .99 / norm
        tau = 1 / sigma_tau_weighting * .99 / norm
        pred_val = chambolle_pock(K, X_val, eta, L, tau, sigma, theta)
        loss_values_val.append(tf.reduce_mean((pred_val - Y_val) ** 2))
        # auxiliary variables for early stopping
        K_best = K.numpy().copy()
        val_best = loss_values_val[0].numpy().copy()
        print(f'0\t\t\t\t\t\t\t{loss_values_val[-1]}')
        # train
        start = time.time()
        for epoch in range(epochs):
            for x, y in ds_tr:
                train_step(x, y)
            # append validation loss
            norm = tf.linalg.norm(K, ord=2)
            sigma = sigma_tau_weighting * .99 / norm
            tau = 1 / sigma_tau_weighting * .99 / norm
            pred_val = chambolle_pock(K, X_val, eta, L, tau, sigma, theta)
            loss_values_val.append(tf.reduce_mean((pred_val - Y_val) ** 2))
            # append train loss
            loss_values.append(train_loss.result().numpy())
            train_loss.reset_states()
            # update best K if indicated
            if loss_values_val[-1] < val_best:
                K_best = K.numpy().copy()
                val_best = loss_values_val[-1].numpy().copy()
            # display losses
            print(f'{epoch+1}\t{loss_values[-1]}\t{loss_values_val[-1]}')
            execution_times.append(time.time() - start)

        # save K
        np.savetxt(os.path.join(log_dir, f'kernel_{specs_string}.txt'), K_best)
        np.savetxt(os.path.join(log_dir, f'loss_values_{specs_string}.txt'), np.array(loss_values))
        np.savetxt(os.path.join(log_dir, f'loss_values_val_{specs_string}.txt'), np.array(loss_values_val))
        np.savetxt(os.path.join(log_dir, f'execution_times_{specs_string}.txt'), np.array(execution_times))
