import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def determine_policy(delay_sum, norm_factor=2.5e5):
    prob_0 = sigmoid(delay_sum / norm_factor - 1)
    prob_1 = prob_0 * sigmoid(delay_sum / norm_factor - 1)
    prob_2 = 1 - prob_0 - prob_1
    return np.random.choice([0, 1, 2], p=[prob_0, prob_1, prob_2])
