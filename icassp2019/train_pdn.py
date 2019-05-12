#!/usr/bin/env python3

"""
train_pdn.py: Train a primal-dual network for dequantization.

Input:
    n - frame size / number of columns in K
    m - dual dimension / number of rows in K
    L - number of unrolled Chambolle-Pock iterations / primal-dual blocks
    E - number of epochs during training
    Delta - quantization interval length
    S_train - features (quantized signals, training set)
    S_dev - features (quantized signals, dev set)
    STilde_train - labels (ground truth signals, training set)
    STilde_dev - labels (ground truth signals, dev set)
    dct_init - toggle for initialization with DCT matrix (optional)
    tau - initial value for step size parameter (optional)
    sigma - initial value for step size parameter (optional)
    learning_rate - learning rate for SGD (optional)
    batch_size - batch size for SGD (optional)
    H_train - inverse weighting filters for training objective (optional)
    H_dev - inverse weighting filters to evaluate objective on dev set (opt.)
    indices_train - indices specifying locations of first frames of signals in
                    the training set (optional)
    indices_dev - indices specifying locations of first frames of signals in
                  the dev set (optional)
    folder - location to save results (optional)
"""

# =============================================================================
# python packages
# =============================================================================
import tensorflow as tf
import numpy as np
import time
from pdn_lib import construct_pdn, create_batch_generator

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
# function definition
# =============================================================================
def train_pdn(n, m, L, E, Delta, S_train, S_dev, STilde_train, STilde_dev,
              dct_init=True, tau=None, sigma=None, learning_rate=1e-4,
              batch_size=512, H_train=None, H_dev=None, indices_train=None,
              indices_dev=None, folder=''):
        
    # auxiliary string for dct initialization
    if dct_init:
        dct_str = '_dct'
    else:
        dct_str = ''
        
    # auxiliary variables for fir filter
    if H_train is not None:
        filter_str = '_filter'
        filter_use = True
    else:
        filter_str = ''
        filter_use = False
        
    # initialize arrays for losses
    training_losses = np.zeros(E + 1)
    dev_losses = np.zeros(E + 1)
    
    # start clock
    t0 = time.time()
    
    # display model information
    print('n=%d\t m=%d\t L=%d'%(n, m, L))
    
    # construct tensorflow graph
    g, S, STilde, H, weights, loss, train = construct_pdn(n, m, L, Delta,
                                                    dct_init=dct_init,
                                                    tau=tau, sigma=sigma,
                                                    learning_rate=learning_rate,
                                                    filter_use=filter_use)
    print('finished graph construction...')
    
    # initialize tensorflow session
    with tf.Session(graph=g) as sess:   
        
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # initialize losses
        training_loss = 0
        dev_loss = 0
        # compute initial losses ...
        if filter_use:
            # ... on training data
            for i in range(1, indices_train.shape[0]):
                # compute current batch corresponding to i-th signal in training set
                S_batch = S_train[:, indices_train[i-1]:indices_train[i]]
                STilde_batch = STilde_train[:, indices_train[i-1]:indices_train[i]]
                H_batch = H_train[:, indices_train[i-1]:indices_train[i]]
                # compute batch loss
                training_loss_batch = sess.run(loss, feed_dict={S:S_batch, STilde:STilde_batch, H:H_batch})
                # update overall loss
                training_loss = training_loss + training_loss_batch
            # average overall training loss
            training_loss = training_loss / indices_train.shape[0]
            # ... on dev data
            for i in range(1, indices_dev.shape[0]):
                # compute current batch corresponding to i-th signal in dev set
                S_batch = S_dev[:, indices_train[i-1]:indices_train[i]]
                STilde_batch = STilde_dev[:, indices_train[i-1]:indices_train[i]]
                H_batch = H_dev[:, indices_train[i-1]:indices_train[i]]
                # compute batch loss
                dev_loss_batch = sess.run(loss, feed_dict={S:S_batch, STilde:STilde_batch, H:H_batch})
                # update overall loss
                dev_loss = dev_loss + dev_loss_batch
            # average overall dev loss
            dev_loss = dev_loss / indices_dev.shape[0]
        else:
            # get overall losses directly (only possible without use of fir filters)
            training_loss = sess.run(loss, feed_dict={S:S_train, STilde:STilde_train})
            dev_loss = sess.run(loss, feed_dict={S:S_dev, STilde:STilde_dev})
        # save and display initial losses
        training_losses[0] = training_loss
        dev_losses[0] = dev_loss
        # compute step sizes after epoch
        tau_learned, sigma_learned = sess.run([weights['tau'], weights['sigma']])
        print('epoch %2d\t %.10f\t %.10f\t %.10f\t %.10f'
              % (0, training_losses[0], dev_losses[0], tau_learned, sigma_learned))
            
        # perform num_epoch epochs
        for epoch in range(1, E+1):
            # initialize losses
            training_loss = 0
            dev_loss = 0
            # perform one epoch on ...
            if filter_use:
                # ... training data (weight updates and loss computation)
                for i in range(1, indices_train.shape[0]):
                    # compute current batch corresponding to i-th signal in training set
                    S_batch = S_train[:, indices_train[i-1]:indices_train[i]]
                    STilde_batch = STilde_train[:, indices_train[i-1]:indices_train[i]]
                    H_batch = H_train[:, indices_train[i-1]:indices_train[i]]
                    # perform gradient step and compute batch loss
                    _, training_loss_batch = sess.run([train, loss], feed_dict={S:S_batch, STilde:STilde_batch, H:H_batch})
                    # update overall loss
                    training_loss = training_loss + training_loss_batch
                # average overall training loss
                training_loss = training_loss / indices_train.shape[0]
                # ... dev data (only loss computation)
                for i in range(1, indices_dev.shape[0]):
                    # compute current batch corresponding to i-th signal in dev set
                    S_batch = S_dev[:, indices_train[i-1]:indices_train[i]]
                    STilde_batch = STilde_dev[:, indices_train[i-1]:indices_train[i]]
                    H_batch = H_dev[:, indices_train[i-1]:indices_train[i]]
                    # compute batch loss
                    dev_loss_batch = sess.run(loss, feed_dict={S:S_batch, STilde:STilde_batch, H:H_batch})
                    # update overall loss
                    dev_loss = dev_loss + dev_loss_batch
                # average overall dev loss
                dev_loss = dev_loss / indices_dev.shape[0]
            else:
                # process training data
                batch_generator = create_batch_generator(S_train, STilde_train, batch_size, shuffle=True)
                # each batch contains random frames from all training signals
                for S_batch, STilde_batch in batch_generator:
                    # perform gradient step
                    sess.run(train, feed_dict={S:S_batch, STilde:STilde_batch})
                # get losses
                training_loss = sess.run(loss, feed_dict={S:S_train, STilde:STilde_train})
                dev_loss = sess.run(loss, feed_dict={S:S_dev, STilde:STilde_dev})
            # compute step sizes after epoch
            tau_learned, sigma_learned = sess.run([weights['tau'], weights['sigma']])
            # save and display final loss at the end of current epoch
            training_losses[epoch] = training_loss
            dev_losses[epoch] = dev_loss
            print('epoch %2d\t %.10f\t %.10f\t %.10f\t %.10f'
                  % (epoch, training_losses[epoch], dev_losses[epoch], tau_learned, sigma_learned))
 
        # stop clock
        t = time.time()-t0
        
        # save weigths
        np.savetxt('%sK%s%s_n%d_m%d_L%d_E%d.txt'%(folder, dct_str, filter_str, n, m, L, E),
                   sess.run(weights['K']))
        np.savetxt('%stau%s%s_n%d_m%d_L%d_E%d.txt'%(folder, dct_str, filter_str, n, m, L, E),
                   np.array([sess.run(weights['tau'])]))
        np.savetxt('%ssigma%s%s_n%d_m%d_L%d_E%d.txt'%(folder, dct_str, filter_str, n, m, L, E),
                   np.array([sess.run(weights['sigma'])]))
        
    # save losses and running time
    np.savetxt('%straining_losses%s%s_n%d_m%d_L%d_E%d.txt' %
               (folder, dct_str, filter_str, n, m, L, E), training_losses)
    np.savetxt('%sdev_losses%s%s_n%d_m%d_L%d_E%d.txt' %
               (folder, dct_str, filter_str, n, m, L, E), dev_losses)
    np.savetxt('%srunning_time%s%s_n%d_m%d_L%d_E%d.txt' %
               (folder, dct_str, filter_str, n, m, L, E), np.array([t]))