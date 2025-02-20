import numpy as np

def relative_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)

def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2

def normalized_squared_error(y_true, y_pred):
    return ((y_true - y_pred) / y_true) ** 2

dict_of_error_functions = {
    "relative_error": relative_error,
    "squared_error": squared_error,
    "normalized_squared_error": normalized_squared_error
}