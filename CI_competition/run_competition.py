import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from CI_competition.data.DataCIModel import DataCIModel
from CI_competition.estimators.models_definitions import models_definitions


def calc_true_ATEs(df):
    """
    :param df: A dataframe including the columns "T" and "Y"
    :return: a dataframe with the ATEs for each treatment
    """
    T_means = df.groupby("T")["Y"].mean()
    treatments = sorted(df["T"].unique().tolist())
    ATEs = {f"T={t}": [] for t in treatments}
    for t in treatments:
        for t2 in treatments:
            ATEs[f"T={t}"].append(T_means[t] - T_means[t2])
    df_ATEs = pd.DataFrame(ATEs, index=[f"T={t}" for t in treatments])
    return df_ATEs


def calc_all_ATEs(args):
    """
    Calculate all the ATEs for the given run, both estimate and True
    """
    simulation_data_dir, index, competition_args, run_args = args

    training_data_filename = run_args["training_data_filename"]
    testing_data_filename = run_args["testing_data_filename"]

    curdir = os.path.dirname(__file__)
    ATEs_dir = os.path.join(curdir, "ATEs")
    os.makedirs(ATEs_dir, exist_ok=True)

    ATEs_dir = os.path.join(ATEs_dir, f"run_{index}")
    os.makedirs(ATEs_dir, exist_ok=True)

    # prepare data
    training_df = pd.read_pickle(f"{simulation_data_dir}/run_{index}/{training_data_filename}.pkl")
    data = DataCIModel(training_df)

    # Run all models
    MODELS_DEFINITIONS = models_definitions()
    if competition_args["model"]:
        models_instances = [MODELS_DEFINITIONS[competition_args["model"]]["class"](**params) for params in
                            MODELS_DEFINITIONS[competition_args["model"]]["params"]]
    else:
        models_instances = [MODELS_DEFINITIONS[model]["class"](**params) for model in MODELS_DEFINITIONS for params in
                            MODELS_DEFINITIONS[model]["params"]]

    for model in models_instances:
        model.estimate_ATE(data).to_pickle(f"{ATEs_dir}/{model.__str__()}.pkl")

    # Calculate the true ATEs
    testing_df = pd.read_pickle(f"{simulation_data_dir}/run_{index}/{testing_data_filename}.pkl")
    calc_true_ATEs(testing_df).to_pickle(f"{ATEs_dir}/True.pkl")


def main(competition_args, run_args):
    indices = range(run_args["continue_from"], run_args["num_runs"] + run_args["continue_from"])
    simulation_data_dir = run_args["simulation_data_dir"]
    with Pool(run_args["num_processes"]) as p:
        list(tqdm(p.map(calc_all_ATEs,
                        [(simulation_data_dir, index, competition_args, run_args) for index in indices]),
                  total=len(indices)))
