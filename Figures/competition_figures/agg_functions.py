import numpy as np
from typing import List

def mean_on_list(list_of_values: List):
    return np.mean(list_of_values)

def median_on_list(list_of_values: List):
    return np.median(list_of_values)

def std_on_list(list_of_values: List):
    return np.std(list_of_values)

def rooted_mean_on_list(list_of_values: List):
    return np.sqrt(np.mean(list_of_values))

dict_of_agg_functions = {
    "Mean": mean_on_list,
    "median": median_on_list,
    "std": std_on_list,
    "rooted_mean": rooted_mean_on_list
}