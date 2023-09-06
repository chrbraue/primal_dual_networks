"""
reproduce_figure_3.py: This script is to reproduce Figure 3 from the EUSIPCO 2023
paper "Asymptotic analysis and truncated backpropagation for the unrolled
primal-dual algorithm"

Input: Make sure to execute run_training.py and run_evaluation.py first (in that
order), and that the respective outputs are in the folder logs. Moreover, take care
that the parameters under (*) and (**) are the same as in run_training.py and
run_evaluation.py.
"""

# =============================================================================
# import python packages
# =============================================================================
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'DejaVu Serif'
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# =============================================================================
# import own code
# =============================================================================
from auxiliary_functions import chambolle_pock

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
# load test data
# =============================================================================
X_test = np.loadtxt('IEEE_corpus/inputs_test.txt')
Y_test = np.loadtxt('IEEE_corpus/outputs_test.txt')

# =============================================================================
# specify hyperparameter values to be considered below (*)
# =============================================================================
L_values = [10, 20, 30, 40, 60, 80, 100, 200, 500, 1000]

# =============================================================================
# specify other parameters that will be kept constant over all below runs (**)
# =============================================================================
bitrate = 4
sigma_tau_weighting = 10
theta = 1

# =============================================================================
# initialize lists to contain plot data
# =============================================================================
loss_full_bp_classical = []
loss_full_bp_optimality = []
loss_truncated_bp_classical = []
loss_truncated_bp_asymptotic = []
execution_times_truncated_bp = []
execution_times_full_bp = []

# =============================================================================
# evaluate models
# =============================================================================
eta = 1 / (2 ** bitrate)

for L in L_values:

    log_dir = f'logs/{L}_iter'

    specs_string = f'truncated_bp_{L}'
    K_truncated_bp = tf.cast(np.loadtxt(os.path.join(log_dir, f'kernel_{specs_string}.txt')), tf.float32)
    norm = tf.linalg.norm(K_truncated_bp, ord=2)
    sigma = sigma_tau_weighting * .99 / norm
    tau = 1 / sigma_tau_weighting * .99 / norm
    pred_val = chambolle_pock(K_truncated_bp, X_test, eta, L, tau, sigma, theta)
    loss_truncated_bp_classical.append(tf.reduce_mean((pred_val - Y_test) ** 2))
    loss_truncated_bp_asymptotic.append(np.loadtxt(os.path.join(log_dir, f'distances_{specs_string}.txt'))[-1])
    execution_times_truncated_bp.append(np.loadtxt(os.path.join(log_dir, f'execution_times_{specs_string}.txt'))[-1])

    specs_string = f'full_bp_{L}'
    K_full_bp = tf.cast(np.loadtxt(os.path.join(log_dir, f'kernel_{specs_string}.txt')), tf.float32)
    norm = tf.linalg.norm(K_full_bp, ord=2)
    sigma = sigma_tau_weighting * .99 / norm
    tau = 1 / sigma_tau_weighting * .99 / norm
    pred_val = chambolle_pock(K_full_bp, X_test, eta, L, tau, sigma, theta)
    loss_full_bp_classical.append(tf.reduce_mean((pred_val - Y_test) ** 2))
    loss_full_bp_optimality.append(np.loadtxt(os.path.join(log_dir, f'distances_{specs_string}.txt'))[-1])
    execution_times_full_bp.append(np.loadtxt(os.path.join(log_dir, f'execution_times_{specs_string}.txt'))[-1])

# =============================================================================
# generate figure
# =============================================================================
label_to_tick = dict(zip(L_values, list(np.arange(15))))
fig = plt.figure(figsize=(7, 4))
[line1] = plt.plot([label_to_tick[label] for label in L_values], loss_full_bp_classical, '--', color='mediumblue')
[line2] = plt.plot([label_to_tick[label] for label in L_values], loss_truncated_bp_classical, '--', color='orange')
[line3] = plt.plot([label_to_tick[label] for label in L_values], loss_full_bp_optimality, '.-', color='mediumblue')
[line4] = plt.plot([label_to_tick[label] for label in L_values], loss_truncated_bp_asymptotic, '.-', color='orange')
ylim = plt.gca().get_ylim()
plt.vlines(6.5, ylim[0], ylim[1], 'k', linewidth=1, alpha=0.5)
plt.vlines(2.5, ylim[0], ylim[1], 'k', linewidth=1, alpha=0.5)
plt.gca().set_ylim(ylim)
plt.xticks(range(len(L_values)))
plt.gca().set_xticklabels(L_values)
plt.xlim([-.5, len(L_values) - .5])
plt.yscale('log')
plt.xlabel('$L_{{Training}}$', fontsize=12)
plt.ylabel('MSE Test', fontsize=12)
plt.gca().tick_params(which='minor', length=0)

width = 0.35
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.bar([label_to_tick[label] - width / 2 for label in L_values], execution_times_truncated_bp, width, color='orange', alpha=0.22)
ax2.bar([label_to_tick[label] + width / 2 for label in L_values], execution_times_full_bp, width, color='mediumblue', alpha=0.22)
ax2.set_yticks([0, 1e4, 2e4, 3e4])
ax2.set_yticklabels(['0', r'$10^4$', r'$2\times 10^4$', r'$3\times 10^4$'], fontdict={'family': 'serif'})
ax2.set_ylabel('Runtime (s)', fontsize=12)

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.set_frame_on(False)

props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
plt.gca().text(0.045, 0.14, 'Regime 1', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
plt.gca().text(0.423, 0.585, 'Regime 2', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
plt.gca().text(0.75, 0.585, 'Regime 3', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)

legend1 = ax1.legend([line1, line2], ['Full BP', 'Truncated BP'], title=f'Classical Inference\n$L_{{Inference}} = L_{{Training}}$', loc='upper center', bbox_to_anchor=(0.345, 1), framealpha=1, fontsize=12)
plt.setp(legend1.get_title(), fontsize=12)
legend2 = ax1.legend([line3, line4], ['Full BP', 'Truncated BP'], title=f'Optimality Inference\n$L_{{Inference}} = 10^{{5}}$', loc='upper right',  bbox_to_anchor=(.933, 1), framealpha=1, fontsize=12)
plt.setp(legend2.get_title(), fontsize=12)
ax1.add_artist(legend1)

plt.tight_layout()
plt.savefig('figure_3.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
