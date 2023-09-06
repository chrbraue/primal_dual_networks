This folder contains code and data to reproduce results from our EUSIPCO 2023 paper "Asymptotic analysis and truncated backpropagation for the unrolled primal-dual algorithm".

To reproduces Figures 3 & 4 from the paper, proceed as follows:

1. Execute fetch_training_data.py to get the training data from icassp2018/IEEE_corpus
2. Execute run_training.py to train unrolling models
3. Execute run_evaluation.py to evaluate trained models performing more iterations than during training
4. Execute reproduce_figure_[3 and 4].py to reproduce the figures from the paper

Feel free to contact us via christoph.brauer@dlr.de.
