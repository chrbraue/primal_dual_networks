"""
reproduce_figure_4.py: This script is to reproduce Figure 3 from the EUSIPCO 2023
paper "Asymptotic analysis and truncated backpropagation for the unrolled
primal-dual algorithm"

Input: Make sure to execute run_training.py and run_evaluation.py first (in that
order), and that the respective outputs are in the folder logs.
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
# authorship information
# =============================================================================
__author__ = 'Christoph Brauer'
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Christoph Brauer'
__email__ = 'christoph.brauer@dlr.de'
__status__ = 'Prototype'

# =============================================================================
# generate figure
# =============================================================================
L_values = [30, 80, 1000]

plt.figure(figsize=(7, 4))
styles = [':', '--', ',-', '-']
plots_truncated_bp = []
plots_full_bp = []
plots_baseline = []

i = 0
for L in L_values:
    log_dir = f'logs/{L}_iter'

    specs_string = f'full_bp_{L}'
    plots_full_bp += plt.plot(np.loadtxt(os.path.join(log_dir, f'distances_{specs_string}.txt')), styles[i], color='mediumblue')

    specs_string = f'truncated_bp_{L}'
    plots_truncated_bp += plt.plot(np.loadtxt(os.path.join(log_dir, f'distances_{specs_string}.txt')), styles[i], color='orange')
    i += 1

# plot baseline (no reconstruction at all)
X_test = tf.cast(np.loadtxt('IEEE_corpus/inputs_test.txt'), tf.float32)
Y_test = tf.cast(np.loadtxt('IEEE_corpus/outputs_test.txt'), tf.float32)
distance = tf.reduce_mean((X_test - Y_test) ** 2)
xlim = plt.gca().get_xlim()
plt.hlines(distance, xlim[0], xlim[1], linestyles='dashed', label='Quantized')

# plot dct curve
distances = np.loadtxt(f'logs/dct/distances.txt')
plots_baseline += plt.plot(distances, 'k-', label='DCT')

labels = [f'$L_{{Training}} = {num_iter}$' for num_iter in L_values]
legend1 = plt.legend(plots_full_bp, labels, title='Full BP', loc='upper right', bbox_to_anchor=(0.6725, .95), framealpha=1, fontsize=12)
plt.setp(legend1.get_title(), fontsize=12)
legend2 = plt.legend(plots_truncated_bp, labels, title='Truncated BP', loc='upper right', bbox_to_anchor=(0.99, .95), framealpha=1, fontsize=12)
plt.setp(legend2.get_title(), fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.235), fontsize=12)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

plt.xlabel('$L_{{Inference}}$', fontsize=12)
plt.ylabel('MSE Test', labelpad=-12, fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.gca().tick_params(which='minor', length=0)
plt.tight_layout()
plt.savefig('figure_4.png', bbox_inches='tight', pad_inches=0.1, dpi=400)
