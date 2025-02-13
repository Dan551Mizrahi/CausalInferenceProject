import numpy as np


def sigmoid(x):
    """ Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def determine_policy(delay_sum, norm_factor=2.5e5):
    """ Determine the policy based on the delay sum"""
    p = sigmoid(55 * delay_sum / norm_factor - 3)
    prob_0 = 1 - p
    prob_1 = p - p ** 2
    prob_2 = p ** 2
    return np.random.choice([0, 1, 2], p=[prob_0, prob_1, prob_2])
