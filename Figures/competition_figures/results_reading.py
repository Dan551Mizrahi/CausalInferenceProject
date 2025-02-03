import pandas as pd
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))


def read_model_est_pickle(model_name: str, run_num: int):
    """
    Read the model estimates from the pickle file
    :param model_name: The name of the model
    :param run_num: The number of the run
    :return: The model estimates
    """
    path_to_results = os.path.join(project_dir, "CI_competition", "ATEs", f"run_{run_num}", f"{model_name}.pkl")
    model_estimates = pd.read_pickle(path_to_results)
    return model_estimates

def read_all_model_estimates(model_name: str, num_runs: int):
    """
    Read all the model estimates from the pickle files
    :param model_name: The name of the model
    :param num_runs: The number of runs
    :return: The model estimations for the first num_runs datasets.
    """
    model_estimations = []
    for run_num in range(num_runs):
        model_estimations.append(read_model_est_pickle(model_name, run_num))
    return model_estimations

def find_all_model_names():
    """
    Find all the model names in the ATEs directory
    :return: The model names
    """
    path_to_results = os.path.join(project_dir, "CI_competition", "ATEs", "run_0")
    model_names = [file_name.split(".")[0] for file_name in os.listdir(path_to_results)]
    return model_names

def read_all_models(num_runs: int):
    """
    Read all the model estimates from the pickle files
    :param num_runs: The number of datasets
    :return: The model estimations for the first num_runs datasets.
    """
    model_estimations = {}
    model_names = find_all_model_names()
    for model_name in model_names:
        model_estimations[model_name] = read_all_model_estimates(model_name, num_runs)
    return model_estimations

def read_true_ates(run_num: int):
    """
    Read the true ATEs from the pickle file
    :param run_num: The number of the datasets
    :return: The true ATEs
    """
    path_to_results = os.path.join(project_dir, "Simulated_Data", f"run_{run_num}", "testing_ATEs_bootstrap.pkl")
    true_ates = pd.read_pickle(path_to_results)
    return true_ates